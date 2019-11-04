/*
Autor: Murillo Freitas Bouzon
Linguagem: C++
Criado em: 14/10/2019
Modificado em: 31/10/2019

Exerc�cio 1 da disciplina PEL 208 - T�picos Especiais em Aprendizagem
Projeto: Regress�o Linear pelo M�todo dos M�nimos Quadrados

*/


#include "LinearRegression.h"

using namespace std;

int main()
{
    //leitura da base
    Matrix M("shoes.csv");
    cout<<M<<"\n";

    //MMC linear
    LinearRegression L(M);
    L.LinearLeastSquares();
    cout<<L.getCoef()<<"\n";
    double X[M.getCols()-1];
    for(int i=0;i<M.getCols()-1;i++)
        cin>>X[i];
    cout<<L.predict(X)<<"\n";

    //MMC quadr�tico
    LinearRegression L2(M);
    L2.QuadraticLeastSquares();
    cout<<L2.getCoef()<<"\n";
    double X2[M.getCols()-1];
    for(int i=0;i<M.getCols()-1;i++)
        cin>>X2[i];
    cout<<L2.predict(X2)<<"\n";

    //MMC robusto
    LinearRegression L3(M);
    L3.RobustLeastSquares();
    cout<<L3.getCoef()<<"\n";
    double X3[M.getCols()-1];
    for(int i=0;i<M.getCols()-1;i++)
        cin>>X3[i];
    cout<<L3.predict(X3)<<"\n";


    return 0;
}
