#define _USE_MATH_DEFINES
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "../functions/data_conversion.h"
#include "../functions/neural_network.h"
#include "../functions/bond.h"
using namespace std;

int main()
{    
    int rows_x = 500;
    int rows_y = 6551;
    int rows_g = 386;
    int cols_x = 14;
    int cols_y = 2;

    double interest_rate;
    int target_duration = 365 * 10;
    int maturity_time_error = 365;
    double default_year = 60;
    int num_companies = 0;

    string** X_string = read_csv("../data/S&P500 Financial Statements_2026-08-01.csv", rows_x+1, 16, 1, 1);
    string** Y_string = read_csv("../data/S&P500 Corporate Bonds_2026-08-01.csv", rows_y+1, 6, 1, 1);
    string** G_string = read_csv("../data/US Government Bond_2026-08-01.csv", rows_g+1, 5, 1, 1);

    double cum_vol = 0.0;
    interest_rate = 0.0;
    for(int r3 = 1; r3 < rows_g+1; r3++)
    {
        double gov_ytm = strtod(&(G_string[r3][1][0]), nullptr);
        double gov_coupon = strtod(&(G_string[r3][2][0]), nullptr);
        int gov_time = strtod(&(G_string[r3][4][0]), nullptr);
        double gov_bond_duration = macaulay_duration(gov_ytm/100.0, gov_coupon/100.0, gov_time/365+1) * 365.0 - (365-gov_time%365);

        if(abs(gov_bond_duration - target_duration) < maturity_time_error)
        {
            double temp_vol = strtod(&(G_string[r3][3][0]), nullptr);
            interest_rate *= cum_vol / (cum_vol + temp_vol);
            interest_rate += gov_ytm * temp_vol / (cum_vol + temp_vol);
            cum_vol += temp_vol;
        }
    }
    cout << interest_rate/100.0 << endl << endl;
    
    string symbols[rows_y];
    double** X = nullptr;
    double** Y = nullptr;
    double* w = nullptr;

    int structure[2] = {5, 2};
    MLP_softmax model(X, Y, structure, 0, cols_x, 2, w);

    for(int r1 = 1; r1 < rows_x+1; r1++)
    {
        bool company_recorded = false;
        double* x = new double[cols_x];
        double total_debt_value = strtod(&X_string[r1][4][0], nullptr);
        x[0] = log(strtod(&X_string[r1][2][0], nullptr)+total_debt_value);
        x[1] = log(strtod(&X_string[r1][3][0], nullptr));
        x[2] = log(strtod(&X_string[r1][5][0], nullptr));
        x[3] = log(strtod(&X_string[r1][6][0], nullptr));
        x[4] = log(strtod(&X_string[r1][7][0], nullptr));
        x[5] = log(strtod(&X_string[r1][8][0], nullptr));
        x[6] = log(strtod(&X_string[r1][9][0], nullptr));
        x[7] = log(strtod(&X_string[r1][10][0], nullptr));
        x[8] = log(strtod(&X_string[r1][9][0], nullptr) + strtod(&X_string[r1][10][0], nullptr));
        x[9] = log(strtod(&X_string[r1][11][0], nullptr)+total_debt_value);
        x[10] = log(strtod(&X_string[r1][12][0], nullptr)+total_debt_value);
        x[11] = log(strtod(&X_string[r1][13][0], nullptr)+total_debt_value);
        x[12] = log(strtod(&X_string[r1][14][0], nullptr)+total_debt_value);
        x[13] = log(strtod(&X_string[r1][15][0], nullptr)+total_debt_value);

        bool all_real_number = true;
        for(int i = 0; i < cols_x; i++)
        {
            x[i] -= log(total_debt_value);
            if(x[i] != x[i])
            {
                all_real_number = false;
                break;
            }
            else
            {
                x[i] = min(max(x[i], -10.0), 10.0);
            }
        }

        for(int r2 = 1; r2 < rows_y+1; r2++)
        {
            bool symbol_matched = true;
            int c;

            for(c = 0; X_string[r1][0][c] != '\0'; c++)
            {
                if(X_string[r1][0][c] != Y_string[r2][0][c])
                {
                    symbol_matched = false;
                    break;
                }
            }
            if(symbol_matched && Y_string[r2][0][c] >= 'A' && Y_string[r2][0][c] <= 'Z')
            {
                symbol_matched = false;
            }

            double ytm = strtod(&Y_string[r2][2][0], nullptr);
            double coupon_rate = strtod(&Y_string[r2][3][0], nullptr);
            int time_to_maturity = strtod(&(Y_string[r2][5][0]), nullptr);
            double corporate_bond_duration = macaulay_duration(ytm/100.0, coupon_rate/100.0, time_to_maturity/365+1) * 365.0 - (365-time_to_maturity%365);

            if(symbol_matched && abs(corporate_bond_duration - target_duration) < maturity_time_error)
            {   
                double* y = new double[cols_y];
                double data_weight = (strtod(&Y_string[r2][4][0], nullptr)+0.0)/1e6;

                y[0] = (1 + interest_rate / 100.0) / (1 + ytm / 100.0);
                y[0] > 1 ? y[0] = 1: y[0] = pow(y[0], default_year);
                y[1] = 1 - y[0];

                if(ytm != ytm)
                {
                    all_real_number = false;
                }

                if(all_real_number)
                {
                    cout << symbols[model.rows] << ' ' << corporate_bond_duration / 365.0 << ' ' << ytm/100.0-interest_rate/100.0 << endl;
                    model.add_one_vector(x, y, data_weight);
                    symbols[model.rows] = Y_string[r2][0];
                    if(!company_recorded)
                    {
                        num_companies += 1;
                    }
                    company_recorded = true;
                }

                delete[] y;
                y = nullptr;
            }
        }
        delete[] x;
        x = nullptr;
    }

    clear_csv(X_string, rows_x+1);
    clear_csv(Y_string, rows_y+1);
    clear_csv(G_string, rows_g+1);

    srand(time(0));
    model.xavier();

    double last_error = model.entropy_loss();
    double diff;
    for(int i = 0; i < 10000; i++)
    {
        model.one_step_fit_backward(0.5);
        diff = last_error - model.entropy_loss();
        if(diff < 1e-6 && diff > -1e-6)
        {
            break;
        }
        last_error = model.entropy_loss();
        cout << i << " : " << last_error << endl;
    }

    double* predicted = new double[model.rows];
    double* actual = new double[model.rows];
    
    for(int i = 0; i < model.rows; i++)
    {
        double* temp_predict = model.predict(model.X[i]);
        predicted[i] = log(1.0 / temp_predict[0]) / default_year;
        actual[i] = log(1.0 / model.Y[i][0]) / default_year;
        cout << symbols[i] << ' ';
        cout << predicted[i] << ' ' << actual[i] << endl;
        delete[] temp_predict;
    }
    double r_square_value = weighted_r_square(predicted, actual, model.rows, model.sample_weights);
    double rho_value = weighted_rho(predicted, actual, model.rows, model.sample_weights);
    cout << "number of bonds: " << model.rows << endl;
    cout << "number of companies:" << num_companies << endl;
    cout << "R2: " << r_square_value << endl;
    cout << "rho: " << rho_value << endl;

    string parameter_file_name = "parameters/neural_network_model_parameters.txt";
    if(r_square_value > 0.6)
    {
        model.store_parameters(&parameter_file_name[0]);
    }

    delete_matrix(model.X, model.rows);
    delete_matrix(model.Y, model.rows);
    delete_vector(model.sample_weights);

    delete[] predicted;
    delete[] actual;

    return 0;
}