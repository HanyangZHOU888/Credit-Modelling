#include <iostream>
#include "statistics.h"
using namespace std;

int main()
{
    cout << norm_inv(normal_dist_cdf(5.9)) << endl;
}