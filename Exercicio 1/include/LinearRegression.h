#include "Matrix.h"

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

//defini��o da classe LinearRegression utilizada para realizar uma regress�o linear
class LinearRegression
{
    private:
        Matrix B; //vetor que armazena os coeficientes betas
        Matrix X; //matriz para representar as vari�veis de entrada
        Matrix Y; //matriz para representar a vari�vel de sa�da
        Matrix data; //matriz com os dados originais da base
    public:
        LinearRegression(Matrix M = Matrix()); //construtor padr�o
        void LinearLeastSquares(); //regress�o linear pelo m�todo dos m�nimos quadrados linear
        void QuadraticLeastSquares(); //regress�o linear pelo m�todo dos m�nimos quadrados quadr�tico
        void RobustLeastSquares(); //regress�o linear pelo m�todo dos m�nimos quadrados robusto
        double predict(double *input); //prediz um valor de y dado um valor de entrada x
        Matrix getCoef(); //retorna os coeficientes da reta calculada
        virtual ~LinearRegression(); //destrutor padr�o
};

#endif // LINEAR_REGRESSION_H
