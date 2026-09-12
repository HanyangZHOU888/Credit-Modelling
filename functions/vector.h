#ifndef _VECTOR_H
#define _VECTOR_H

#include <iostream>
using namespace std;

void display_matrix(double** A, int m, int n)
{
    for(int r = 0; r < m; r++)
    {
        for(int c = 0; c < n; c++)
        {
            cout << A[r][c] << ' ';
        }
        cout << endl;
    }
    cout << endl;
}

void display_vector(double* v, int n)
{
    for(int i = 0; i < n; i++)
    {
        cout << v[i] << ' ';
    }
    cout << endl;
}

double** create_matrix(int m, int n)
{
    double** result = new double*[m];
    for(int i = 0; i < m; i++)
    {
        result[i] = new double[n];
    }
    return result;
}

void delete_matrix(double**& A, int m)
{
    for(int i = 0; i < m; i++)
    {
        if(A[i] != nullptr)
        {
            delete[] A[i];
            A[i] = nullptr;
        }
    }
    if(A != nullptr)
    {
        delete[] A;
        A = nullptr;
    }
}

void copy_vector(double* dest, double* source, int n)
{
    for(int i = 0; i < n; i++)
    {
        dest[i] = source[i];
    }
}

void set_vector(double* v, int n, double value)
{
    for(int i = 0; i < n; i++)
    {
        v[i] = value;
    }
}

double** add_vector_to_matrix(double**& A, double*& a, int m, int n)
{
    int batch = 500;
    double** result = A;
    if(m % batch == 0)
    {
        result = new double* [m+batch];
        for(int i = 0; i < m; i++)
        {
            result[i] = new double[n];
            copy_vector(result[i], A[i], n);
        }
        if(A != nullptr)
        {
            delete_matrix(A, m);
        }
    }
    result[m] = new double[n];
    copy_vector(result[m], a, n);

    delete[] a;
    a = nullptr;

    return result; 
}

double* add_num_to_vector(double*& v, double value, int m)
{
    double* result = new double[m+1];
    copy_vector(result, v, m);
    result[m] = value;
    if(v != nullptr)
    {
        delete[] v;
    }
    v = nullptr;
    return result;
}

void delete_vector(double*& v)
{
    delete[] v;
    v = nullptr;
}

#endif