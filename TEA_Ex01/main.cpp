/*
Autor: Murillo Freitas Bouzon
Linguagem: C++
Criado em: 14/10/2019
Modificado em: 17/10/2019

Exercício 1 da disciplina PEL 208 - Tópicos Especiais em Aprendizagem
Projeto: Regressão Linear pelo Método dos Mínimos Quadrados

*/


#include "Matrix.h"

using namespace std;

int main()
{
    Matrix M(2, 2);
    cin>>M;
    cout<<M.inverse()<<"\n";
    return 0;
}
