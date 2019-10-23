#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
using namespace std;

#ifndef MATRIX_H
#define MATRIX_H

//Definição da classe Matrix para representar uma matriz de números reais de tamanho RxC
class Matrix{
    private:
        int R; //Número de linhas
        int C; //Número de colunas
        double **M; //Matriz onde são guardados os elementos
    public:
        Matrix(); //construtor padrão
        ~Matrix(); //destrutor padrão
        Matrix(int N,int M, double **data = nullptr); //construtor que recebe a quantidade de linhas, colunas
        Matrix(string F); //construtor que le a matriz a partir de um arquivo
        Matrix operator +(const Matrix M2); //sobrecarga do operador +
        Matrix operator *(const Matrix M2); //sobrecarga do operador *
        Matrix operator *(double k); //sobrecarga do operador +
        Matrix operator +(double k); //sobrecarga do operador *
        double operator [] (pair<int,int> P); //sobrecarga do operador []
        friend ostream& operator << (ostream& out, Matrix A); //sobrecarga do operador <<
        friend istream& operator >> (istream& input, Matrix &A); //sobrecarga do operador >>
        double determinante(); //método para calcular o determinante da matriz
        Matrix CofMatrix(int N,int M,int x, int y = 0); //método para calcular a matriz de cofator x,y
        Matrix transpose(); //método para calcular a transposta da matriz
        Matrix inverse(); // método para calcular a inversa da matriz
        void fillMatrix(double x); //método para preencher a matriz com um valor x
        int getRows(); //método que retorna o número de linhas
        int getCols(); //método que retorna o número de colunas

};


#endif
