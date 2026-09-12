#ifndef _MATRIX_ALGEBRA_H
#define _MATRIX_ALGEBRA_H

#include <iostream>
#include "vector.h"
using namespace std;

void vector_plus(double* v1, double* v2, int length)
{
    for(int i = 0; i < length; i++)
    {
        v1[i] += v2[i];
    }
}

void vector_minus(double* v1, double* v2, int length)
{
    for(int i = 0; i < length; i++)
    {
        v1[i] -= v2[i];
    }
}

void vector_multiply(double* v1, double k, int length)
{
    for(int i = 0; i < length; i++)
    {
        v1[i] *= k;
    }
}

void vector_divide(double* v1, double k, int length)
{
    for(int i = 0; i < length; i++)
    {
        v1[i] /= k;
    }
}

void vector_swap(double* v1, double* v2, int length)
{
    double temp;
    for(int k = 0; k < length; k++)
    {
        temp = v1[k]; *(v1+k) = v2[k]; *(v2+k) = temp;
    }
}

double dot_product(double* v1, double* v2, int length)
{
    double result = 0.0;

    for(int i = 0; i < length; i++)
    {
        result += v1[i] * v2[i];
    }

    return result;
}

double* cross_product(double** A, double* x, int m, int n)
{
    double* result = new double[m];
    for(int i = 0; i < m; i++)
    {
        result[i] = dot_product(A[i], x, n);
    }
    return result;
}

void matrix_solve(double** A, double* b, int n)
{
    for(int i = 0; i < n; i++)
    {
        if(A[i][i] == 0)
        {
            for(int r = i+1; r < n; r++)
            {
                if(A[r][i] != 0)
                {
                    vector_swap(A[i], A[r], n);
                    double temp = b[i];
                    b[i] = b[r];
                    b[r] = temp;
                    break;
                }
            }
        }

        if(A[i][i] == 0)
        {
            cerr << "Warning: Matrix is not invertible" << endl;
            return;
        }

        double factor = 1 / A[i][i];
        b[i] *= factor;
        vector_multiply(A[i], factor, n);

        for(int r = i+1; r < n; r++)
        {
            if(A[r][i] != 0.0)
            {
                factor = A[i][i]/A[r][i];
                vector_multiply(A[r], factor, n);
                b[r] *= factor;
                vector_minus(A[r], A[i], n);
                b[r] -= b[i];
            }
        }
    }

    for(int i = n-2; i >= 0; i--)
    {
        for(int j = n-1; j > i; j--)
        {
            b[i] -= A[i][j] * b[j];
            A[i][j] = 0;
        }
    }
}

double* least_square(double** A, double* b, int m, int n)
{
    double* v1 = new double[m];
    double* v2 = new double[m];

    double** temp_a = create_matrix(n, n);
    double* temp_b = new double[n];
    for(int r = 0; r < n; r++)
    {
        for(int i = 0; i < m; i++)
        {
            v1[i] = A[i][r];
        }
        for(int c = 0; c < n; c++)
        {
            for(int j = 0; j < m; j++)
            {
                v2[j] = A[j][c];
            }
            temp_a[r][c] = dot_product(v1, v2, m);
        }
        temp_b[r] = dot_product(v1, b, m);
    }

    matrix_solve(temp_a, temp_b, n);
    
    delete_vector(v1);
    delete_vector(v2);
    delete_matrix(temp_a, n);
    return temp_b;
}

#endif