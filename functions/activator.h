#ifndef _ACTIVATOR_H
#define _ACTIVATOR_H

#include <cmath>
#include "linear_algebra.h"

double sigmoid(double x)
{
    x > 30.0 ? x = 30.0 : x = x;
    x < -30.0 ? x = -30.0 : x = x;
    return 1 / (1 + exp(-x));
}

double sigmoid(double* x, double* w, double constant, int length)
{
    return 1 / (1 + exp(-dot_product(x, w, length)-constant));
}

double linear(double* x, double* w, double constant, int length)
{
    return dot_product(x, w, length) + constant;
}
#endif