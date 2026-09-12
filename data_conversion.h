#ifndef _DATA_CONVERSION
#define _DATA_CONVERSION

#include <string>
#include <cstring>
#include <fstream>
using namespace std;

double string_to_double(char* s)
{
    double num = 0.0;
    bool dotted = false;
    int digit = 1;
    for(int i = 0; s[i] != '\0' && s[i] != '\n' && s[i] != ' '; i++)
    {
        if(s[i] >= '0' && s[i] <= '9')
        {
            if(dotted == false)
            {
                num *= 10;
                num += s[i] - '0';
            }
            else
            {
                double temp = (s[i] - '0') + 0.0;
                for(int c = 0; c < digit; c++)
                {
                    temp /= 10;
                }
                num += temp;
                digit += 1;
            }
        }
        else if(s[i] == '.')
        {
            dotted = true;
        }
    }

    if(s[0] == '-')
    {
        num = -num;
    }

    return num;
}

string** read_csv(string filename, int rows, int cols, int row_batch = 1, int grid_batch = 1)
{
    string** result = nullptr;
    int char_id_start;

    ifstream csv_file(filename);
    if (!csv_file.is_open())
    {
        cerr << "Can't open the file " << filename << endl;
        return nullptr;
    }

    string temp_line;
    for (int r = 0; r < rows && getline(csv_file, temp_line); r++)
    {
        // 行扩容
        if (r % row_batch == 0)
        {
            string** temp_result = new string*[r + row_batch];
            for (int i = 0; i < r; i++)
            {
                temp_result[i] = result[i];
            }
            if (result != nullptr)
            {
                delete[] result;
            }
            result = temp_result;
        }
        result[r] = new string[cols];

        char_id_start = 0;
        for (int c = 0; c < cols; c++)
        {
            char* position = nullptr;
            int grid_length = 0;

            for (int w = char_id_start;; w++)
            {
                // 扩容
                if (grid_length % grid_batch == 0)
                {
                    char* temp_position = new char[grid_length + grid_batch];
                    for (int k = 0; k < grid_length; k++)
                    {
                        temp_position[k] = position[k];
                    }
                    if (position != nullptr)
                    {
                        delete[] position;
                    }
                    position = temp_position;
                }

                // 到达行末尾，终止
                if (w >= temp_line.size())
                {
                    position[grid_length] = '\0';
                    char_id_start = w + 1;
                    break;
                }

                if (temp_line[w] != ',')
                {
                    position[grid_length++] = temp_line[w];
                }
                else
                {
                    position[grid_length] = '\0';
                    char_id_start = w + 1;
                    break;
                }
            }

            // 修复：直接赋值std::string，禁止strcpy
            result[r][c] = position;

            // 释放临时char数组，解决内存泄漏
            delete[] position;
            position = nullptr;
        }
    }

    csv_file.close();
    return result;
}

void clear_csv(string** csv_file, int rows)
{
    for(int r = 0; r < rows; r++)
    {
        if(csv_file[r] != nullptr)
        {
            delete[] csv_file[r];
            csv_file[r] = nullptr;
        }
    }
    delete[] csv_file;
    csv_file = nullptr;
}

#endif