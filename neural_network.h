#ifndef _NEURAL_NETWORK_H
#define _NEURAL_NETWORK_H
#define _USE_MATH_DEFINES
#include <cmath>

#include "activator.h"
#include "vector.h"
#include "calculus.h"
#include "statistics.h"
#include "data_conversion.h"

class MLP_softmax
{
    public:
        double** X;
        double** Y;
        double* sample_weights;
        int rows;
        int cols;

        int num_of_layers;
        int* structure_of_layers;
        int num_of_outputs;
        double sum_sample_w;

        double*** weights;

        MLP_softmax(double** x, double** y, int* structure, int r, int c, int n, double* sample_w)
        {
            X = x;
            Y = y;
            structure_of_layers = structure;
            rows = r;
            cols = c;
            num_of_layers = n;
            sample_weights = sample_w;
            num_of_outputs = structure_of_layers[num_of_layers-1];
            sum_sample_w = sum(sample_weights, rows);

            int num_of_neuron_last_layer;

            weights = new double**[num_of_layers];
            for(int i = 0; i < num_of_layers; i++)
            {
                if(i == 0)
                {
                    num_of_neuron_last_layer = cols;
                }
                else
                {
                    num_of_neuron_last_layer = structure_of_layers[i-1];
                }
                weights[i] = new double*[structure_of_layers[i]];
                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    weights[i][j] = new double[num_of_neuron_last_layer+1];
                }
            }
        }

        void add_one_vector(double* x, double* y, double w = 1.0)
        {
            if(rows % 500 == 0)
            {
                double** new_X = new double*[(rows+500)];
                double** new_Y = new double*[(rows+500)];
                double* new_sw = new double[(rows+500)];
                for(int r = 0; r < rows; r++)
                {
                    new_X[r] = new double[cols];
                    copy_vector(new_X[r], X[r], cols);
                    new_Y[r] = new double[num_of_outputs];
                    copy_vector(new_Y[r], Y[r], num_of_outputs);
                    new_sw[r] = sample_weights[r];
                }
                
                if(X != nullptr)
                {
                    delete_matrix(X, rows);
                }
                if(Y != nullptr)
                {
                    delete_matrix(Y, rows);
                }
                if(sample_weights != nullptr)
                {
                    delete_vector(sample_weights);
                }
                
                X = new_X; Y = new_Y; sample_weights = new_sw;
                for(int r = rows; r < rows+500; r++)
                {
                    new_X[r] = nullptr;
                    new_Y[r] = nullptr;
                    new_sw[r] = 0.0;
                }
            }
            
            X[rows] = new double[cols];
            copy_vector(X[rows], x, cols);
            Y[rows] = new double[num_of_outputs];
            copy_vector(Y[rows], y, num_of_outputs);
            sample_weights[rows] = w;
            rows += 1;
            sum_sample_w += w;
        }

        void xavier()
        {
            for(int i = 0; i < num_of_layers; i++)
            {
                if(i == 0)
                {
                    for(int j = 0; j < structure_of_layers[i]; j++)
                    {
                        for(int k = 0; k < cols+1; k++)
                        {
                            weights[i][j][k] = sqrt(6.0 / (cols + structure_of_layers[i])) * ((2.0 * rand()) / RAND_MAX - 1.0);
                        }
                    }
                }
                else
                {
                    for(int j = 0; j < structure_of_layers[i]; j++)
                    {
                        for(int k = 0; k < structure_of_layers[i-1]+1; k++)
                        {
                            weights[i][j][k] = sqrt(6.0 / (structure_of_layers[i-1] + structure_of_layers[i])) * ((2.0 * rand()) / RAND_MAX - 1.0);
                        }
                    }
                }
            }
        }

        void store_parameters(char* filename)
        {
            ofstream trun(filename, ios::trunc);
            trun.close();

            ofstream fout;
            fout.open(filename);

            int num_of_neuron_last_layer = cols;
            for(int i = 0; i < num_of_layers; i++)
            {
                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    for(int k = 0; k < num_of_neuron_last_layer+1; k++)
                    {
                        fout << weights[i][j][k] << endl;
                    }
                }
                num_of_neuron_last_layer = structure_of_layers[i];
            }

            fout.close();
        }

        void get_parameters(char* filename)
        {
            ifstream fin;
            fin.open(filename);
            
            string buffer;
            int num_of_neuron_last_layer = cols;
            for(int i = 0; i < num_of_layers; i++)
            {
                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    for(int k = 0; k < num_of_neuron_last_layer+1; k++)
                    {
                        getline(fin, buffer);
                        weights[i][j][k] = std::stod(buffer);
                    }
                }
                num_of_neuron_last_layer = structure_of_layers[i];
            }

            fin.close();
        }

        void display_weights()
        {
            int num_of_neuron_last_layer = cols;
            for(int i = 0; i < num_of_layers; i++)
            {
                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    for(int k = 0; k < num_of_neuron_last_layer+1; k++)
                    {
                        cout << weights[i][j][k] << ' ';
                    }
                }
                num_of_neuron_last_layer = structure_of_layers[i];
            }
            cout << endl;
        }

        double* predict(double* x)
        {
            double* result = new double[num_of_outputs];
            double** output = new double*[num_of_layers];
            double* input = x;;
            int input_dim = cols;

            for(int i = 0; i < num_of_layers; i++)
            {
                output[i] = new double[structure_of_layers[i]];
                if(i == num_of_layers - 1)
                {
                    for(int j = 0; j < structure_of_layers[i]; j++)
                    {
                        output[i][j] = dot_product(input, weights[i][j], input_dim) + weights[i][j][input_dim];
                    }
                    double maximum = output[i][0];
                    for(int j = 0; j < structure_of_layers[i]; j++)
                    {
                        if(output[i][j] > maximum)
                        {
                            maximum = output[i][j];
                        }
                    }
                    double sum = 0.0;
                    for(int j = 0; j < structure_of_layers[i]; j++)
                    {
                        output[i][j] = exp(output[i][j]-maximum);
                        sum += output[i][j];
                    }
                    for(int j = 0; j < structure_of_layers[i]; j++)
                    {
                        output[i][j] /= sum;
                    }
                }
                else
                {
                    for(int j = 0; j < structure_of_layers[i]; j++)
                    {
                        output[i][j] = sigmoid(input, weights[i][j], weights[i][j][input_dim], input_dim);
                    }
                }
                input = output[i];
                input_dim = structure_of_layers[i];
            }
            
            copy_vector(result, output[num_of_layers-1], structure_of_layers[num_of_layers-1]);
            delete_matrix(output, num_of_layers);
            return result;
        }

        double entropy_loss()
        {
            double result = 0.0;
            double* predicted;
            double sum_w = 0.0;
            double a;
            double y;
            for(int i = 0; i < rows; i++)
            {
                double sample_loss = 0.0;
                predicted = predict(X[i]);
                for(int j = 0; j < num_of_outputs; j++)
                {
                    a = predicted[j];
                    y = Y[i][j];
                    a > 1-dx ? a = 1-dx : a = a;
                    a < dx ? a = dx : a = a;
                    sample_loss -= y * log(a);
                }
                result *= sum_w / (sum_w+sample_weights[i]);
                result += sample_loss * (sample_weights[i] / (sum_w+sample_weights[i]));
                sum_w += sample_weights[i];
                delete_vector(predicted);
            }
            return result;
        }

        void one_step_fit_backward(double alpha)
        {
            // 1. 初始化梯度矩阵，维度和weights完全一致，初始化为0
            double*** dw = new double** [num_of_layers];
            for (int l = 0; l < num_of_layers; l++)
            {
                int input_dim = (l == 0) ? cols : structure_of_layers[l - 1];
                dw[l] = new double* [structure_of_layers[l]];
                for (int j = 0; j < structure_of_layers[l]; j++)
                {
                    dw[l][j] = new double[input_dim + 1](); // () 表示初始化为0
                }
            }

            // 2. 遍历每个样本，累加梯度
            for (int sample_idx = 0; sample_idx < rows; sample_idx++)
            {
                double* x = X[sample_idx];
                double* y = Y[sample_idx];
                double sw_i = sample_weights[sample_idx];

                // ---------- 2.1 前向传播：保存每一层的激活值 ----------
                double** a = new double* [num_of_layers];
                for (int l = 0; l < num_of_layers; l++)
                {
                    a[l] = new double[structure_of_layers[l]];
                    int input_dim = (l == 0) ? cols : structure_of_layers[l - 1];
                    double* input_vec = (l == 0) ? x : a[l - 1];

                    if(l == num_of_layers - 1)
                    {
                        // ========== 输出层：Softmax（数值稳定版） ==========
                        double* z = new double[structure_of_layers[l]];
                        for(int j = 0; j < structure_of_layers[l]; j++)
                        {
                            z[j] = dot_product(input_vec, weights[l][j], input_dim) 
                                + weights[l][j][input_dim];
                        }
                        double max_z = z[0];
                        for(int j = 1; j < structure_of_layers[l]; j++)
                        {
                            if(z[j] > max_z) max_z = z[j];
                        }
                        double sum_exp = 0.0;
                        for(int j = 0; j < structure_of_layers[l]; j++)
                        {
                            a[l][j] = exp(z[j] - max_z);
                            sum_exp += a[l][j];
                        }
                        for(int j = 0; j < structure_of_layers[l]; j++)
                        {
                            a[l][j] /= sum_exp;
                            // 保留数值裁剪，避免 log(0) 导致损失无穷大
                            a[l][j] = std::max(dx, std::min(1.0 - dx, a[l][j]));
                        }
                        delete[] z;
                    }
                    else
                    {
                        // ========== 隐藏层：保持 Sigmoid 不变 ==========
                        for (int j = 0; j < structure_of_layers[l]; j++)
                        {
                            double bias = weights[l][j][input_dim];
                            a[l][j] = sigmoid(input_vec, weights[l][j], bias, input_dim);
                            a[l][j] = std::max(dx, std::min(1.0 - dx, a[l][j]));
                        }
                    }
                }

                // ---------- 2.2 计算每一层的误差项 delta ----------
                double** delta = new double* [num_of_layers];
                for (int l = 0; l < num_of_layers; l++)
                {
                    delta[l] = new double[structure_of_layers[l]];
                }

                // 输出层误差：delta = a - y （交叉熵+Sigmoid的化简结果）
                int output_layer = num_of_layers - 1;
                for (int j = 0; j < num_of_outputs; j++)
                {
                    delta[output_layer][j] = a[output_layer][j] - y[j];
                }

                // 隐藏层误差：从倒数第二层向前反向传递
                for (int l = num_of_layers - 2; l >= 0; l--)
                {
                    int next_layer = l + 1;
                    for (int j = 0; j < structure_of_layers[l]; j++)
                    {
                        // 误差反向加权求和：后一层误差 * 对应权重
                        double error_sum = 0.0;
                        for (int k = 0; k < structure_of_layers[next_layer]; k++)
                        {
                            // weights[next_layer][k][j] = 后一层第k个神经元的第j个输入权重
                            error_sum += delta[next_layer][k] * weights[next_layer][k][j];
                        }
                        // 乘以Sigmoid导数 a*(1-a)
                        delta[l][j] = error_sum * a[l][j] * (1.0 - a[l][j]);
                    }
                }

                // ---------- 2.3 计算当前样本的梯度，累加到dw中 ----------
                for (int l = 0; l < num_of_layers; l++)
                {
                    int input_dim = (l == 0) ? cols : structure_of_layers[l - 1];
                    double* input_act = (l == 0) ? x : a[l - 1]; // 前一层的激活值

                    for (int j = 0; j < structure_of_layers[l]; j++)
                    {
                        // 普通权重的梯度：delta * 前层激活
                        for (int k = 0; k < input_dim; k++)
                        {
                            dw[l][j][k] += sw_i * delta[l][j] * input_act[k];
                        }
                        // 偏置的梯度：等于delta本身（对应权重数组最后一位）
                        dw[l][j][input_dim] += sw_i * delta[l][j];
                    }
                }

                // 释放当前样本的临时数组
                delete_matrix(a, num_of_layers);
                delete_matrix(delta, num_of_layers);
            }

            // 3. 梯度取平均（所有样本的平均损失的梯度
            for (int l = 0; l < num_of_layers; l++)
            {
                int input_dim = (l == 0) ? cols : structure_of_layers[l - 1];
                for (int j = 0; j < structure_of_layers[l]; j++)
                {
                    for (int k = 0; k < input_dim + 1; k++)
                    {
                        dw[l][j][k] /= sum_sample_w;
                    }
                }
            }

            // 4. 梯度下降更新权重
            for (int l = 0; l < num_of_layers; l++)
            {
                int input_dim = (l == 0) ? cols : structure_of_layers[l - 1];
                for (int j = 0; j < structure_of_layers[l]; j++)
                {
                    for (int k = 0; k < input_dim + 1; k++)
                    {
                        weights[l][j][k] -= alpha * dw[l][j][k];
                    }
                }
            }

            // 5. 释放梯度矩阵
            for (int l = 0; l < num_of_layers; l++)
            {
                for (int j = 0; j < structure_of_layers[l]; j++)
                {
                    delete[] dw[l][j];
                }
                delete[] dw[l];
            }
            delete[] dw;
        }

        void one_step_fit_forward(double alpha)
        {
            double*** dw = new double**[num_of_layers];
            int num_of_neuron_last_layer;

            for(int i = 0; i < num_of_layers; i++)
            {
                dw[i] = new double*[structure_of_layers[i]];

                if(i == 0)
                {
                    num_of_neuron_last_layer = cols;
                }
                else
                {
                    num_of_neuron_last_layer = structure_of_layers[i-1];
                }

                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    dw[i][j] = new double[num_of_neuron_last_layer+1];
                }
            }

            double temp = entropy_loss();
            double de_dw;
            for(int i = 0; i < num_of_layers; i++)
            {
                if(i == 0)
                {
                    num_of_neuron_last_layer = cols;
                }
                else
                {
                    num_of_neuron_last_layer = structure_of_layers[i-1];
                }

                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    for(int k = 0; k < num_of_neuron_last_layer+1; k++)
                    {
                        weights[i][j][k] += 1e-6;
                        de_dw = (entropy_loss() - temp) / dx;
                        dw[i][j][k] = - alpha * de_dw;
                        weights[i][j][k] -= dx;
                    }
                }
            }

            for(int i = 0; i < num_of_layers; i++)
            {
                if(i == 0)
                {
                    num_of_neuron_last_layer = cols;
                }
                else
                {
                    num_of_neuron_last_layer = structure_of_layers[i-1];
                }

                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    for(int k = 0; k < num_of_neuron_last_layer+1; k++)
                    {
                        weights[i][j][k] += dw[i][j][k];
                    }
                }
            }

            for(int i = 0; i < num_of_layers; i++)
            {
                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    delete[] dw[i][j];
                    dw[i][j] = nullptr;
                }
                delete[] dw[i];
                dw[i] = nullptr;
            }
            delete[] dw;
            dw = nullptr;
        }

        ~MLP_softmax()
        {
            for(int i = 0; i < num_of_layers; i++)
            {
                for(int j = 0; j < structure_of_layers[i]; j++)
                {
                    delete[] weights[i][j];
                    weights[i][j] = nullptr;
                }
                delete[] weights[i];
                weights[i] = nullptr;
            }
            delete[] weights;
            weights = nullptr;
        }
};

#endif