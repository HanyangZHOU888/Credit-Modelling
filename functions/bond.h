#ifndef _BOND_H
#define _BOND_H
#include <cmath>
#include "../statistics.h"
#include "../linear_algebra.h"

double macaulay_duration(double ytm, double coupon_rate, int t)
{
    double result = 0.0;
    double present_value = 0.0;
    for(int i = 1; i < t+1; i++)
    {
        present_value += coupon_rate / pow(1+ytm, i);
    }
    present_value += 1 / pow(1+ytm, t);
    for(int i = 1; i < t+1; i++)
    {
        result += (coupon_rate / pow(1+ytm, i)) * i / present_value;
    }
    result += 1 / pow(1+ytm, t) * t / present_value;
    return result / (1+ytm);
}

double mertons_model(double asset, double debt, double v, double t, double mu, double r)
{
    double d1 = (log(asset/debt)+(mu+v*v/2)*t) / (v*sqrt(t));
    double d2 = (log(asset/debt)+(mu-v*v/2)*t) / (v*sqrt(t));

    return asset * exp((mu-r)*t) * normal_dist_cdf(-d1) + debt * exp(-r*t) * normal_dist_cdf(d2);
}

double bond_asset_delta(double asset, double debt, double v, double t, double mu, double r)
{
    double d1 = (log(asset/debt)+(mu+v*v/2)*t) / (v*sqrt(t));
    double d2 = (log(asset/debt)+(mu-v*v/2)*t) / (v*sqrt(t));

    return normal_dist_cdf(-d1) * exp((mu-r)*t);
}

double bond_debt_delta(double asset, double debt, double v, double t, double mu, double r)
{
    double d1 = (log(asset/debt)+(mu+v*v/2)*t) / (v*sqrt(t));
    double d2 = (log(asset/debt)+(mu-v*v/2)*t) / (v*sqrt(t));

    return normal_dist_cdf(d2) * exp(-r*t);
}

double bond_vega(double asset, double debt, double v, double t, double mu, double r)
{
    double d1 = (log(asset/debt)+(mu+v*v/2)*t) / (v*sqrt(t));

    return asset * exp((mu-r)*t) * (exp(-0.5*d1*d1) / sqrt(2.0*M_PI)) * sqrt(t);
}

class Linear_Merton_Model
{
    public:
        double** X_asset;
        double** X_debt;
        double* y;
        int m;
        int asset_n;
        int debt_n;

        double* asset_weights;
        double* debt_weights;
        double* data_weights;
        double r;
        double t;
        double volatility;

        Linear_Merton_Model(double** X_asset_in, double** X_debt_in, double* y_in, double* data_weights_in, int m_in, int asset_n_in, int debt_n_in, double r_in, double t_in)
        {
            X_asset = X_asset_in;
            X_debt = X_debt_in;
            y = y_in;
            data_weights = data_weights_in;
            m = m_in;
            asset_n = asset_n_in;
            debt_n = debt_n_in;

            asset_weights = new double[asset_n+1];
            debt_weights = new double[debt_n+1];
            for(int i = 0; i < asset_n+1; i++)
            {
                asset_weights[i] = -1.0;
            }
            for(int i = 0; i < debt_n+1; i++)
            {
                debt_weights[i] = -1.0;
            }

            r = r_in;
            t = t_in;
            volatility = 0.2;
        }

        void fit(int max_iter, double alpha)
        {
            double* asset_derivatives = new double[asset_n+1];
            double* debt_derivatives = new double[debt_n+1];
            double* asset_transformed_weights = new double[asset_n+1];
            double* debt_transformed_weights = new double[debt_n+1];
            double volatility_derivative;
            
            double asset_est;
            double debt_est;
            double d1;
            double d2;

            double last_error;
            for(int i = 0; i < max_iter; i++)
            {
                last_error = error();
                cout << i << " : " << last_error << endl;
                set_vector(asset_derivatives, asset_n+1, 0.0);
                set_vector(debt_derivatives, debt_n+1, 0.0);
                volatility_derivative = 0.0;

                for(int n1 = 0; n1 < asset_n+1; n1++){asset_transformed_weights[n1] = exp(asset_weights[n1]);}
                for(int n2 = 0; n2 < debt_n+1; n2++){debt_transformed_weights[n2] = exp(debt_weights[n2]);}

                double cum_weight = 0.0;

                for(int j = 0; j < m; j++)
                {
                    asset_est = dot_product(asset_transformed_weights, X_asset[j], asset_n)+asset_transformed_weights[asset_n];
                    debt_est = dot_product(debt_transformed_weights, X_debt[j], debt_n)+debt_transformed_weights[debt_n];

                    d1 = (log(asset_est/debt_est) + (r+volatility*volatility/2) * t) / (volatility * sqrt(t));
                    d2 = (log(asset_est/debt_est) + (r-volatility*volatility/2) * t) / (volatility * sqrt(t));

                    double discount = exp(-r*t);
                    double delta_s = normal_dist_cdf(-d1);
                    double delta_k = normal_dist_cdf(d2);

                    /*if(i == 0 || i == max_iter-1)
                    {
                        cout << "asset/debt: " << asset_est/debt_est << ", r: " << r << ", volatility: " << volatility << ", d1: " << d1 << ", d2: " << d2 << ", delta_s: " << delta_s << ", delta_k: " << delta_k << endl;
                    }*/

                    double predicted_pv = asset_est * delta_s + debt_est * discount * delta_k;
                    double predicted = predicted_pv / (debt_est * discount);
                    //double error = -log(predicted)/t + log(y[j])/t;
                    double error = d2 - norm_inv(y[j]);

                    //dy_dasset = (-(1/predicted)*(1.0/t)*(delta_s/(debt_est*exp(-r*t))));
                    double dy_dasset = (1.0 / (volatility * sqrt(t) * asset_est));
                    //double dy_ddebt = (-(1/predicted)*(1.0/t)*((delta_k-predicted)/debt_est));
                    double dy_ddebt = (-1.0 / (volatility * sqrt(t) * debt_est));

                    for(int n1 = 0; n1 < asset_n; n1++)
                    {
                        asset_derivatives[n1] *= cum_weight / (cum_weight + data_weights[j]);
                        asset_derivatives[n1] += alpha * error * dy_dasset * X_asset[j][n1] * exp(asset_weights[n1]) * (data_weights[j] / (cum_weight + data_weights[j]));
                    }
                    asset_derivatives[asset_n] *= cum_weight / (cum_weight + data_weights[j]);
                    asset_derivatives[asset_n] += alpha * error * dy_dasset * exp(asset_weights[asset_n]) * (data_weights[j] / (cum_weight + data_weights[j]));
                    for(int n2 = 0; n2 < debt_n; n2++)
                    {
                        debt_derivatives[n2] *= cum_weight / (cum_weight + data_weights[j]);
                        debt_derivatives[n2] += alpha * error * dy_ddebt * X_debt[j][n2] * exp(debt_weights[n2]) * (data_weights[j] / (cum_weight + data_weights[j]));
                    }
                    debt_derivatives[debt_n] *= cum_weight / (cum_weight + data_weights[j]);
                    debt_derivatives[debt_n] += alpha * error * dy_ddebt * exp(debt_weights[debt_n]) * (data_weights[j] / (cum_weight + data_weights[j]));

                    double vega = -asset_est * sqrt(t) * normal_dist_pdf(d1);
                    volatility_derivative *= (cum_weight / (data_weights[j]+cum_weight));
                    volatility_derivative += alpha * error * (-d1/volatility) /*(-(1/predicted)/t * vega / (debt_est * exp(-r*t)))*/ * (data_weights[j] / (data_weights[j]+cum_weight));

                    cum_weight += data_weights[j];
                }

                vector_minus(asset_weights, asset_derivatives, asset_n+1);
                vector_minus(debt_weights, debt_derivatives, debt_n+1);
                volatility *= exp(-volatility_derivative);
                
                //display_vector(asset_weights, asset_n);
                //display_vector(debt_weights, debt_n);
                //cout << endl;
/*
                bool all_derivatives_zero = true;
                for(int n = 0; n < asset_n+1; n++)
                {
                    if(abs(asset_derivatives[n]) > 1e-6)
                    {
                        all_derivatives_zero = false;
                    }
                }
                for(int n = 0; n < debt_n+1; n++)
                {
                    if(abs(debt_derivatives[n]) > 1e-6)
                    {
                        all_derivatives_zero = false;
                    }
                }
                if(abs(volatility_derivative) > 1e-6)
                {
                    all_derivatives_zero = false;
                }
*/
                if(abs(last_error - error()) < 1e-9)
                {
                    break;
                }
                else
                {
                    last_error = error();
                }
            }
/*
            for(int n1 = 0; n1 < asset_n; n1++)
            {
                asset_weights[n1] = min(20.0, max(-20.0, asset_weights[n1]));
            }
            for(int n2 = 0; n2 < debt_n; n2++)
            {
                debt_weights[n2] = min(20.0, max(-20.0, debt_weights[n2]));
            }
*/
            delete[] asset_transformed_weights;
            delete[] debt_transformed_weights;
            delete[] asset_derivatives;
            delete[] debt_derivatives;
        }

        double predict(double* asset_x, double* debt_x)
        {
            double* asset_transformed_weights = new double[asset_n+1];
            double* debt_transformed_weights = new double[debt_n+1];
            for(int i = 0; i < asset_n+1; i++){asset_transformed_weights[i] = exp(asset_weights[i]);}
            for(int i = 0; i < debt_n+1; i++){debt_transformed_weights[i] = exp(debt_weights[i]);}

            double asset_est = dot_product(asset_x, asset_transformed_weights, asset_n)+asset_transformed_weights[asset_n];
            double debt_est = dot_product(debt_x, debt_transformed_weights, debt_n)+debt_transformed_weights[debt_n];
            double d1 = (log(asset_est/debt_est) + (r+0.5*volatility*volatility)*t) / (volatility*sqrt(t));
            double d2 = (log(asset_est/debt_est) + (r-0.5*volatility*volatility)*t) / (volatility*sqrt(t));
            double pv = /*asset_est * normal_dist_cdf(-d1) + */debt_est * exp(-r*t) * normal_dist_cdf(d2);
            delete[] asset_transformed_weights;
            delete[] debt_transformed_weights;
            return pv / (debt_est * exp(-r*t));
        }

        double error()
        {
            double result = 0.0;
            double cum_weight = 0.0;
            double predicted;
            for(int i = 0; i < m; i++)
            {
                predicted = predict(X_asset[i], X_debt[i]);
                result *= cum_weight / (data_weights[i]+cum_weight);
                result += (y[i] * log(predicted) + (1.0-y[i]) * (1-predicted)) * data_weights[i] / (data_weights[i]+cum_weight);

                cum_weight += data_weights[i];
            }
            return -result;
        }

        ~Linear_Merton_Model()
        {
            delete_vector(asset_weights);
            delete_vector(debt_weights);
        }
};

#endif