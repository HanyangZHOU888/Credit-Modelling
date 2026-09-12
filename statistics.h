#ifndef _STATISTICS_H
#define _STATISTICS_H
#include <cmath>

double mean(double* x, int n)
{
    double result = 0.0;
    for(int i = 0; i < n; i++)
    {
        result += x[i] / n;
    }
    return result;
}

double median(double* x, int n)
{
    double sorted[n];
    double temp;
    for(int i = 0; i < n; i++)
    {
        sorted[i] = x[i];
    }
    for(int i = 0; i < n; i++)
    {
        for(int j = i; j < n; j++)
        {
            if(sorted[i] < sorted[j])
            {
                temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }
    return (sorted[(n-1)/2] + sorted[n/2]) / 2.0;
}

double variance(double* x, int n)
{
    double result = 0.0;
    double m = mean(x, n);
    for(int i = 0; i < n; i++)
    {
        result += (x[i] - m) * (x[i] - m);
    }
    result /= (n-1);
    return result;
}

double sum_abs(double* x, int n)
{
    double result = 0.0;
    for(int i = 0; i < n; i++)
    {
        result += abs(x[i]);
    }
    return result;
}

double sum(double* x, int n)
{
    double result = 0.0;
    for(int i = 0; i < n; i++)
    {
        result += x[i];
    }
    return result;
}

double r_square(double* predicted, double* actual, int n)
{
    double sum_error = 0.0;
    for(int i = 0; i < n; i++)
    {
        sum_error += (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
    }
    sum_error /= n;

    return 1 - sum_error / variance(actual, n);
}

double rho(double* predicted, double* actual, int n)
{
    double sum_error = 0.0;
    double predicted_mean = mean(predicted, n);
    double actual_mean = mean(actual, n);
    for(int i = 0; i < n; i++)
    {
        sum_error += (predicted[i] - predicted_mean) * (actual[i] - actual_mean);
    }
    sum_error /= n;

    return sum_error / sqrt(variance(predicted, n)) / sqrt(variance(actual, n));
}

double weighted_mean(double* x, int n, double* w)
{
    double weights_sum = 0.0;
    double result = 0.0;
    for(int i = 0; i < n; i++)
    {
        result *= weights_sum / (weights_sum + w[i]);
        result += x[i] * w[i] / (weights_sum + w[i]);
        weights_sum += w[i];
    }
    return result;
}

double weighted_variance(double* x, int n, double* w)
{
    double weighted_average = weighted_mean(x, n, w);
    double result = 0.0;
    double weights_sum = 0.0;
    for(int i = 0; i < n; i++)
    {
        result *= weights_sum / (weights_sum + w[i]);
        result += w[i] / (weights_sum + w[i]) * (x[i] - weighted_average) * (x[i] - weighted_average);
        weights_sum += w[i];
    }
    return result;
}

double weighted_r_square(double* predicted, double* actual, int n, double* w)
{
    double sum_error = 0.0;
    double weights_sum = 0.0;
    for(int i = 0; i < n; i++)
    {
        sum_error *= weights_sum / (weights_sum + w[i]);
        sum_error += w[i] / (weights_sum+w[i]) * (predicted[i] - actual[i]) * (predicted[i] - actual[i]);
        weights_sum += w[i];
    }
    return 1 - sum_error / weighted_variance(actual, n, w);
}

double weighted_rho(double* predicted, double* actual, int n, double* w)
{
    double predicted_mean = weighted_mean(predicted, n, w);
    double actual_mean = weighted_mean(actual, n, w);
    double weights_sum = 0.0;
    double sum_error = 0.0;
    for(int i = 0; i < n; i++)
    {
        sum_error *= weights_sum / (weights_sum+w[i]);
        sum_error += w[i] / (weights_sum+w[i]) * (actual[i] - actual_mean) * (predicted[i] - predicted_mean);
        weights_sum += w[i];
    }
    return sum_error / sqrt(weighted_variance(predicted, n, w)) / sqrt(weighted_variance(actual, n, w));
}

double normal_dist_pdf(double z)
{
    return (1 / sqrt(2 * M_PI)) * exp(-0.5 * z * z);
}

double normal_dist_cdf(double z)
{   
    if(z > 6.0)
    {
        return 1;
    }
    if(z < -6.0)
    {
        return 0;
    }
    double sum = 0.0;
    double addition = z;
    for(double i = 1.0; ; i++)
    {
        sum += addition;
        addition *= z;
        addition /= i;
        addition *= z;
        addition /= -2.0;
        addition *= (2.0*i-1); 
        addition /= (2.0*i+1);
        if(addition < 1e-9 && addition > -1e-9)
        {
            break;
        }
    }
    sum /= sqrt(2.0*M_PI);
    return sum + 0.5;
}

double norm_inv(double p)
{
    if(p >= 1.0)
    {
        return 6.0;
    }
    if(p <= 0.0)
    {
        return -6.0;
    }
    double lower = -6.0;
    double upper = 6.0;
    double middle;
    double predicted;
    while(true)
    {
        middle = (lower+upper)/2.0;
        predicted = normal_dist_cdf(middle);
        if(predicted > p + 1e-9)
        {
            upper = middle;
        }
        else if(predicted < p - 1e-9)
        {
            lower = middle;
        }
        else
        {
            break;
        }
    }
    return middle;
}

#endif