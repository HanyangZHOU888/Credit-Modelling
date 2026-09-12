#ifndef _CALCULUS_H
#define _CALCULUS_H

#include "vector.h"
#include "linear_algebra.h"

double dx = 1e-6;
int MAX_ITER = 1e6;

double derivative(double (*func)(double*), double* x, int i, int n)
{
    double temp_x[n];
    copy_vector(temp_x, x, n);
    temp_x[i] += dx;
    
    double result = (func(temp_x) - func(x)) / dx;
    return result;
}

double* partial_derivative(double (*func)(double*), double* x, int n)
{
    double* result = new double[n];

    double temp_x[n];
    copy_vector(temp_x, x, n);

    for(int i = 0; i < n; i++)
    {
        temp_x[i] += dx;
        result[i] = (func(temp_x) - func(x)) / dx;
        temp_x[i] -= dx;
    }

    return result;
}

double** hessian_matrix(double (*func)(double*), double* x, int n)
{
    double** result = create_matrix(n, n);

    double temp_x[n];
    copy_vector(temp_x, x, n);

    for(int i = 0; i < n; i++)
    {
        double first_derivative = derivative(func, x, i, n);
        for(int j = 0; j < n; j++)
        {
            temp_x[j] += dx;
            result[i][j] = (derivative(func, temp_x, i, n) - first_derivative) / dx;
            temp_x[j] -= dx;
        }
    }

    return result;
}

void one_step_gradient_descent(double (*func)(double*), double* x, int n)
{
    double** hessian;
    double* dy_dx_v;
    double* dx_v;

    hessian = hessian_matrix(func, x, n);
    dy_dx_v = partial_derivative(func, x, n);
    vector_multiply(dy_dx_v, -1, n);

    dx_v = least_square(hessian, dy_dx_v, n, n);
    vector_multiply(dx_v, 0.63, n);

    vector_plus(x, dx_v, n);
    
    delete_matrix(hessian, n);
    delete_vector(dy_dx_v);
    delete_vector(dx_v);

    return;
}

void gradient_descent(double (*func)(double*), double*x, int n, bool display_x)
{
    double** hessian;
    double* dy_dx_v;
    double* dx_v;
    bool condition = true;

    for(int i = 0; i < MAX_ITER && condition; i++)
    {
        if(display_x)
        {
            display_vector(x, n);
        }
        hessian = hessian_matrix(func, x, n);
        dy_dx_v = partial_derivative(func, x, n);
        vector_multiply(dy_dx_v, -1, n);

        dx_v = least_square(hessian, dy_dx_v, n, n);
        vector_multiply(dx_v, 0.63, n);

        vector_plus(x, dx_v, n);

        condition = false;
        for(int c = 0; c < n; c++)
        {
            if(dx_v[c] > dx || dx_v[c] < -dx)
            {
                condition = true;
            }
        }

        delete_matrix(hessian, n);
        delete_vector(dy_dx_v);
        delete_vector(dx_v);
    }
}

#endif