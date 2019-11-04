#include "Matrix.h"

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

//definição da classe LinearRegression utilizada para realizar uma regressão linear
class LinearRegression
{
    private:
        Matrix B; //vetor que armazena os coeficientes betas
        Matrix X; //matriz para representar as variáveis de entrada
        Matrix Y; //matriz para representar a variável de saída
        Matrix data; //matriz com os dados originais da base
    public:
        LinearRegression(Matrix M = Matrix()); //construtor padrão
        void LinearLeastSquares(); //regressão linear pelo método dos mínimos quadrados linear
        void QuadraticLeastSquares(); //regressão linear pelo método dos mínimos quadrados quadrático
        void RobustLeastSquares(); //regressão linear pelo método dos mínimos quadrados robusto
        double predict(double *input); //prediz um valor de y dado um valor de entrada x
        Matrix getCoef(); //retorna os coeficientes da reta calculada
        virtual ~LinearRegression(); //destrutor padrão
};

#endif // LINEAR_REGRESSION_H
