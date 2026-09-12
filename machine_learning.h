#ifndef _MACHINE_LEARNING_H
#define _MACHINE_LEARNING_H

#include <cmath>
#include <iostream>

#include "calculus.h"
#include "statistics.h"

class Machine_Learning_Models
{
    public:
        double** A = nullptr;
        double* b = nullptr;
        int m = 0;
        int n = 0;
        double* data_weights = nullptr;

        double* w = nullptr;

        Machine_Learning_Models(double** A_in, double* b_in, int m_in, int n_in, double* data_weights_in)
        {
            A = A_in; b = b_in; m = m_in; n = n_in; data_weights = data_weights_in;
        }

        ~Machine_Learning_Models()
        {
            delete_vector(w);
        }
};

class Linear_Regression : public Machine_Learning_Models
{
    public:
        double R2;
        double e2;
        double std_e;

        Linear_Regression(double** A_in, double* b_in, int m_in, int n_in, double* data_weights_in) : Machine_Learning_Models(A_in, b_in, m_in, n_in, data_weights_in)
        {
            w = least_square(A, b, m, n);

            e2 = e_square();
            std_e = sqrt(e2);
            R2 = 1 - e2 / variance(b, m);
        }

        double predict(double* x)
        {
            return dot_product(w, x, n);
        }
        
        double e_square()
        {
            double sum = 0.0;
            for(int i = 0; i < m; i++){sum += (predict(A[i]) - b[i]) * (predict(A[i]) - b[i]) / m;}
            return sum;
        }

        double de_dw(int i)
        {
            double temp_e = e_square();
            w[i] += dx;
            double result = (e_square() - temp_e) / dx;
            w[i] -= dx;
            return result;
        }
};

class Logistic_Regression : public Machine_Learning_Models
{
    public:
        Logistic_Regression(double** A_in, double* b_in, int m_in, int n_in, double* data_weights_in) : Machine_Learning_Models(A_in, b_in, m_in, n_in, data_weights_in)
        {
            /*
            double temp_b[m];
            for(int i = 0; i < m; i++)
            {
                temp_b[i] = -log(1 / b[i] - 1);
            }

            w = least_square(A, temp_b, m, n);*/

            w = new double[n];
            for(int i = 0; i < n; i++)
            {
                w[i] = 0.0;
            }

            double* dy_dx_v = nullptr;
            bool condition;

            double current_error = entropy_loss();
            for(int i = 0; i < MAX_ITER; i++)
            {
                dy_dx_v = new double[n];
                for(int j = 0; j < n; j++)
                {
                    dy_dx_v[j] = de_dw(j);
                }

                vector_multiply(dy_dx_v, 0.1, n);
                vector_minus(w, dy_dx_v, n);
                display_vector(w, n);
                cout << entropy_loss() << endl;

                if(abs(current_error - entropy_loss()) < 0.00001)
                {                   
                    delete_vector(dy_dx_v);
                    break;
                }

                delete_vector(dy_dx_v);
                current_error = entropy_loss();
            }
        }

        double predict(double* x)
        {
            return 1.0 / (1.0 + exp(0.0-dot_product(x, w, n)));
        }

        double entropy_loss()
        {
            double sum = 0.0;
            double sum_weights = 0.0;
            for(int i = 0; i < m; i++)
            {
                double current_weight = data_weights[i];
                sum *= sum_weights / (sum_weights + current_weight);
                sum += current_weight / (sum_weights + current_weight) * (0.0 - b[i]*log(predict(A[i])) - (1-b[i])*log(1-predict(A[i])));
                sum_weights += current_weight;
            }
            return sum;
        }

        double de_dw(int i)
        {
            double temp_e = entropy_loss();
            w[i] += dx;
            double result = (entropy_loss() - temp_e) / dx;
            w[i] -= dx;
            return result;
        }
};

class Probit_Regression : public Machine_Learning_Models
{
    public:
        Probit_Regression(double** A_in, double* b_in, int m_in, int n_in, double* data_weights_in) : Machine_Learning_Models(A_in, b_in, m_in, n_in, data_weights_in)
        {
            w = new double[n];
            for(int i = 0; i < n; i++)
            {
                w[i] = 0.0;
            }

            double* de_dw = new double[n];

            double current_error = entropy_loss();
            double sum_weights;
            for(int t = 0; t < MAX_ITER; t++)
            {
                cout << t << endl;
                for(int j = 0; j < n; j++)
                {
                    de_dw[j] = 0.0;
                }
                sum_weights = 0.0;
                for(int i = 0; i < m; i++)
                {
                    double predicted = predict(A[i]);
                    double z = dot_product(A[i], w, n);
                    double base = (predicted - b[i]) * (exp(-z*z/2.0) / sqrt(2 * M_PI));
                    double current_weight = data_weights[i];
                    for(int j = 0; j < n; j++)
                    {
                        de_dw[j] *= sum_weights / (sum_weights+current_weight);
                        de_dw[j] += current_weight / (sum_weights+current_weight) * (base * A[i][j]);
                    }
                    sum_weights += current_weight;
                }

                vector_multiply(de_dw, 0.1, n);
                vector_minus(w, de_dw, n);

                if(abs(entropy_loss()-current_error) < 1e-6)
                {
                    break;
                }
                current_error = entropy_loss();
            }      

            delete_vector(de_dw);
        }

        double predict(double* x)
        {
            return normal_dist_cdf(dot_product(w, x, n));
        }

        double entropy_loss()
        {
            double sum = 0.0;
            double sum_weights = 0.0;
            for(int i = 0; i < m; i++)
            {
                double current_weight = data_weights[i];
                sum *= sum_weights / (sum_weights + current_weight);
                sum += current_weight / (sum_weights + current_weight) * (0.0 - b[i] * log(predict(A[i])) - (1-b[i]) * log(1.0-predict(A[i])));
                sum_weights += current_weight;
            }
            return sum;
        }

        double mean_squared_error()
        {
            double sum = 0.0;
            double sum_weights = 0.0;
            for(int i = 0; i < m; i++)
            {
                double current_weight = data_weights[i];
                sum *= sum_weights / (current_weight + sum_weights);
                sum += current_weight / (current_weight + sum_weights) * (b[i]-predict(A[i])) * (b[i]-predict(A[i]));
                sum_weights += current_weight;
            }
            return sum;
        }
};

#endif