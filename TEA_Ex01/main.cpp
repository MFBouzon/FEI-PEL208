/*
Autor: Murillo Freitas Bouzon
Linguagem: C++
Criado em: 14/10/2019
Modificado em: 17/10/2019

Exerc�cio 1 da disciplina PEL 208 - T�picos Especiais em Aprendizagem
Projeto: Regress�o Linear pelo M�todo dos M�nimos Quadrados

*/


#include "LinearRegression.h"

using namespace std;

int main()
{
    Matrix M("Books_attend_grade.csv");
    cout<<M<<"\n";
    LinearRegression L(M);
    L.QuadraticLeastSquares();
    cout<<L.getCoef()<<"\n";
    double X[M.getCols()-1];
    for(int i=0;i<M.getCols()-1;i++)
        cin>>X[i];
    cout<<L.predict(X);
    return 0;
}
