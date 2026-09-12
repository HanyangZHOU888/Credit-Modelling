#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "../functions/data_conversion.h"
#include "../functions/neural_network.h"
using namespace std;

int main()
{    
    int rows_x = 500;
    int cols_x = 13;
    int rows_y = 6551;
    int cols_y = 2;

    double interest_rate = 4.387;
    int maturity_time = 365 * 10;
    int maturity_time_error = 365;
    double default_year = 60;

    string** X_string = read_csv("../data/S&P500 Financial Statements_2026-08-01.csv", rows_x+1, 16, 1, 1);
    string** Y_string = read_csv("../data/S&P500 Corporate Bonds_2026-08-01.csv", rows_y+1, 5, 1, 1);
    
    string symbols[rows_x];
    double** X = nullptr;
    double** Y = nullptr;
    double* w = nullptr;

    int structure[2] = {10, 2};
    MLP_softmax model(X, Y, structure, 0, cols_x, 2, w);

    for(int r1 = 1; r1 < rows_x+1; r1++)
    {
        int total_weight = 0;
        double average_ytm = 0.0;
        bool bond_found = false;

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

            if(symbol_matched && strtod(&(Y_string[r2][4][0]), nullptr) - maturity_time < maturity_time_error && strtod(&(Y_string[r2][4][0]), nullptr) - maturity_time > -maturity_time_error)
            {
                bond_found = true;
                double current_weight = strtod(&Y_string[r2][4][0], nullptr);
                double current_ytm = strtod(&Y_string[r2][2][0], nullptr);
                average_ytm = average_ytm * (total_weight / (total_weight + current_weight)) + current_ytm * (current_weight / (total_weight + current_weight));
                total_weight += current_weight;
            }
        }

        if(bond_found)
        {
            symbols[model.rows] = X_string[r1][0];

            double* x = new double[cols_x];
            double* y = new double[cols_y];

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

            y[0] = pow((1 + interest_rate / 100.0) / (1 + average_ytm / 100.0), default_year);
            y[1] = 1 - y[0];

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
                model.add_one_vector(x, y, 1.0);
            }

            delete[] x;
            x = nullptr;
            delete[] y;
            y = nullptr;
        }
    }

    clear_csv(X_string, rows_x+1);
    clear_csv(Y_string, rows_y+1);

    srand(time(0));
    model.xavier();

    double last_error = model.entropy_loss();
    double diff;
    for(int i = 0; i < 100000; i++)
    {
        cout << i << ":" << last_error << endl;;
        model.one_step_fit_backward(0.5);
        diff = last_error - model.entropy_loss();
        if(diff < 1e-6 && diff > -1e-6)
        {
            break;
        }
        last_error = model.entropy_loss();
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
    double r_square_value = r_square(predicted, actual, model.rows);
    double rho_value = rho(predicted, actual, model.rows);
    cout << "R2: " << r_square_value << endl;
    cout << "rows: " << model.rows << endl;

    string parameter_file_name = "neural_network_model_parameters.txt";
    if(r_square_value > 0.599185)
    {
        model.store_parameters(&parameter_file_name[0]);
    }

    delete_matrix(model.X, model.rows);
    delete_matrix(model.Y, model.rows);

    delete[] predicted;
    delete[] actual;

    return 0;
}