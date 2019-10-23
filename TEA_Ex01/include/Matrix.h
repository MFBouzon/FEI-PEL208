#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
using namespace std;

#ifndef MATRIX_H
#define MATRIX_H

//Defini��o da classe Matrix para representar uma matriz de n�meros reais de tamanho RxC
class Matrix{
    private:
        int R; //N�mero de linhas
        int C; //N�mero de colunas
        double **M; //Matriz onde s�o guardados os elementos
    public:
        Matrix(); //construtor padr�o
        ~Matrix(); //destrutor padr�o
        Matrix(int N,int M, double **data = nullptr); //construtor que recebe a quantidade de linhas, colunas
        Matrix(string F); //construtor que le a matriz a partir de um arquivo
        Matrix operator +(const Matrix M2); //sobrecarga do operador +
        Matrix operator *(const Matrix M2); //sobrecarga do operador *
        Matrix operator *(double k); //sobrecarga do operador +
        Matrix operator +(double k); //sobrecarga do operador *
        double operator [] (pair<int,int> P); //sobrecarga do operador []
        friend ostream& operator << (ostream& out, Matrix A); //sobrecarga do operador <<
        friend istream& operator >> (istream& input, Matrix &A); //sobrecarga do operador >>
        double determinante(); //m�todo para calcular o determinante da matriz
        Matrix CofMatrix(int N,int M,int x, int y = 0); //m�todo para calcular a matriz de cofator x,y
        Matrix transpose(); //m�todo para calcular a transposta da matriz
        Matrix inverse(); // m�todo para calcular a inversa da matriz
        void fillMatrix(double x); //m�todo para preencher a matriz com um valor x
        int getRows(); //m�todo que retorna o n�mero de linhas
        int getCols(); //m�todo que retorna o n�mero de colunas

};


#endif
