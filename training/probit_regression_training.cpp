#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include "../functions/data_conversion.h"
#include "../functions/machine_learning.h"
#include "../functions/bond.h"
using namespace std;

int main()
{    
    int rows_x = 500;
    int cols_x = 15;
    int rows_y = 6551;
    int cols_y = 2;
    int rows_g = 386;

    int num_bonds = 0;
    int num_companies = 0;

    double interest_rate;
    int maturity_time = 365 * 10;
    int maturity_time_error = 365;
    double default_year = 60;

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

        if(abs(gov_bond_duration - maturity_time) < maturity_time_error)
        {
            double temp_vol = strtod(&(G_string[r3][3][0]), nullptr);
            interest_rate *= cum_vol / (cum_vol + temp_vol);
            interest_rate += gov_ytm * temp_vol / (cum_vol + temp_vol);
            cum_vol += temp_vol;
        }
    }
    clear_csv(G_string, rows_g+1);
    
    string symbols[rows_y];
    double** X = nullptr;
    double* Y = nullptr;
    double* data_w = nullptr;

    for(int r1 = 1; r1 < rows_x+1; r1++)
    {
        bool company_found = false;
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

            double weight = strtod(&Y_string[r2][4][0], nullptr)/(0.0+1e3);
            double ytm = strtod(&Y_string[r2][2][0], nullptr);
            double coupon_rate = strtod(&Y_string[r2][3][0], nullptr);
            int time_to_maturity = strtod(&(Y_string[r2][5][0]), nullptr);
            double corporate_bond_duration = macaulay_duration(ytm/100.0, coupon_rate/100.0, time_to_maturity/365+1) * 365.0 - (365-time_to_maturity%365);

            if(symbol_matched && abs(corporate_bond_duration - maturity_time) < maturity_time_error)
            {
                symbols[num_bonds] = Y_string[r2][0];

                double* x = new double[cols_x];
                double y;

                double total_debt_value = strtod(&X_string[r1][4][0], nullptr);
                x[0] = log(strtod(&X_string[r1][2][0], nullptr)+total_debt_value) - log(1e9);
                x[1] = log(strtod(&X_string[r1][3][0], nullptr)) - log(1e9);
                x[2] = log(strtod(&X_string[r1][5][0], nullptr)) - log(1e9);
                x[3] = log(strtod(&X_string[r1][6][0], nullptr)) - log(1e9);
                x[4] = log(strtod(&X_string[r1][7][0], nullptr)) - log(1e9);
                x[5] = log(strtod(&X_string[r1][8][0], nullptr)) - log(1e9);
                x[6] = log(strtod(&X_string[r1][9][0], nullptr)) - log(1e9);
                x[7] = log(strtod(&X_string[r1][10][0], nullptr)) - log(1e9);
                x[8] = log(strtod(&X_string[r1][9][0], nullptr) + strtod(&X_string[r1][10][0], nullptr)) - log(1e9);
                x[9] = log(strtod(&X_string[r1][11][0], nullptr)+total_debt_value) - log(1e9);
                x[10] = log(strtod(&X_string[r1][12][0], nullptr)+total_debt_value) - log(1e9);
                x[11] = log(strtod(&X_string[r1][13][0], nullptr)+total_debt_value) - log(1e9);
                x[12] = log(strtod(&X_string[r1][14][0], nullptr)+total_debt_value) - log(1e9);
                x[13] = log(strtod(&X_string[r1][15][0], nullptr)+total_debt_value) - log(1e9);
                x[14] = 1.0;

                y = max(min(pow((1 + interest_rate / 100.0) / (1 + ytm / 100.0), default_year), 1.0-1e-6), 1e-6);

                bool all_real_number = true;
                for(int i = 0; i < cols_x; i++)
                {
                    x[i] -= log(total_debt_value) - log(1e9);
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

                if(all_real_number)
                {
                    cout << symbols[num_bonds] << ' ' << corporate_bond_duration / 365.0 << ' ' << ytm/100.0-interest_rate/100.0 << endl;
                    X = add_vector_to_matrix(X, x, num_bonds, cols_x);
                    Y = add_num_to_vector(Y, y, num_bonds);
                    data_w = add_num_to_vector(data_w, weight, num_bonds);
                    company_found = true;
                    num_bonds += 1;
                }
                else
                {
                    delete[] x;
                    x = nullptr;
                }
            }
        }
        if(company_found)
        {
            num_companies += 1;
        }
    }

    clear_csv(X_string, rows_x+1);
    clear_csv(Y_string, rows_y+1);

    for(int i = 0; i < num_bonds; i++)
    {
        cout << symbols[i] << " : " << endl;
        for(int j = 0; j < cols_x; j++)
        {
            cout << X[i][j] << ' ';
        }
        cout << endl << Y[i] << endl;
    }
    cout << "\n";

    Probit_Regression model(X, Y, num_bonds, cols_x, data_w);

    double* predicted = new double[num_bonds];
    double* actual = new double[num_bonds];
    
    for(int i = 0; i < num_bonds; i++)
    {
        predicted[i] = log(1.0 / model.predict(X[i])) / default_year;
        actual[i] = log(1.0 / Y[i]) / default_year;

        cout << symbols[i] << endl;
        cout << ' ' << predicted[i] << ' ' << actual[i] << endl;
    }
    double r_square_value = weighted_r_square(predicted, actual, num_bonds, data_w);
    double rho_value = weighted_rho(predicted, actual, num_bonds, data_w);
    cout << "R2: " << r_square_value << endl;
    cout << "rho: " << rho_value << endl;
    cout << "number of bonds: " << num_bonds << endl;
    cout << "number of companues: " << num_companies << endl;

    delete_matrix(X, num_bonds);
    delete_vector(Y);
    delete_vector(data_w);

    delete[] predicted;
    delete[] actual;

    return 0;
}