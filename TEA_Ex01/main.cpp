/*
Autor: Murillo Freitas Bouzon
Linguagem: C++
Criado em: 14/10/2019
Modificado em: 17/10/2019

Exerc�cio 1 da disciplina PEL 208 - T�picos Especiais em Aprendizagem
Projeto: Regress�o Linear pelo M�todo dos M�nimos Quadrados

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
