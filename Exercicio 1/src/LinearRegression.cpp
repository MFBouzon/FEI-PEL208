#include "LinearRegression.h"

//construtor que recebe os dados representados em forma matricial dada pela Matrix M
LinearRegression::LinearRegression(Matrix M)
{
    data = M;
    B = Matrix(0,0);
}

//regressão linear pelo método dos mínimos quadrados linear
void LinearRegression::LinearLeastSquares()
{

    int R = data.getRows();
    int C = data.getCols();

    double **out = new double *[R];
    double **A;

    A = new double*[R];
    for(int i=0;i<R;i++){
        A[i] = new double[C];
        for(int j=0;j<C;j++){
            if(j == 0)
                A[i][j] = 1;

            else
                A[i][j] = data[make_pair(i,j-1)];
        }
    }

    for(int i=0;i<R;i++){
        out[i] = new double;
        out[i][0] = data[make_pair(i,C-1)];
    }

    X = Matrix(R, C, A);
    Y = Matrix(R, 1, out);

    delete A;
    delete out;

    B = (X.transpose()*X).inverse()* X.transpose() * Y;
}

//regressão linear pelo método dos mínimos quadrados de forma quadrática
void LinearRegression::QuadraticLeastSquares()
{

    int R = data.getRows();
    int C = 1 + (data.getCols()-1)*2;

    double **out = new double *[R];
    double **A;

    A = new double*[R];

    for(int i=0;i<R;i++){
        A[i] = new double[C];
        int ind = 0;
        for(int j=0;j<C;j++){
            if(j == 0)
                A[i][j] = 1;
            else if(j % 2 != 0)
                A[i][j] = data[make_pair(i,ind)];
            else
                A[i][j] = data[make_pair(i,ind)]* data[make_pair(i,ind++)];
        }
    }

    for(int i=0;i<R;i++){
        out[i] = new double;
        out[i][0] = data[make_pair(i,data.getCols()-1)];
    }

    X = Matrix(R, C, A);
    Y = Matrix(R, 1, out);


    delete A;
    delete out;

    B = (X.transpose()*X).inverse()* X.transpose() * Y;

}

//regressão linear pelo método dos mínimos quadrados ponderado
void LinearRegression::RobustLeastSquares()
{
    if(B.getCols() == 0)
        LinearLeastSquares();

    int R = data.getRows();
    int C = data.getCols();

    double **out = new double *[R];
    double **A;
    double W[R];
    double in[C-1];

    A = new double*[R];
    for(int i=0;i<R;i++){
        A[i] = new double[C];
        int a = 0;
        for(int j=0;j<C;j++){
            if(j == 0)
                A[i][j] = 1;

            else{
                A[i][j] = data[make_pair(i,j-1)];
                in[a++] =A[i][j];
            }
        }
        W[i] = predict(in);
    }

    for(int i=0;i<R;i++){
        out[i] = new double;
        out[i][0] = data[make_pair(i,C-1)];
        W[i] = 1/abs(out[i][0] - W[i]);
    }

    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            A[i][j] = W[i]*A[i][j];
        }
    }

    for(int i=0;i<R;i++){
        out[i][0] = W[i]*out[i][0];
    }

    X = Matrix(R, C, A);
    Y = Matrix(R, 1, out);

    cout<<"\n"<<X<<"\n";


    delete A;
    delete out;

    cout<<"\n"<<(X.transpose()*X)<<"\n";
    cout<<"\n"<<(X.transpose()*X).inverse()<<"\n";

    B = (X.transpose()*X).inverse()* X.transpose() * Y;

}

//prediz um valor de saída dado um valor de entrada
double LinearRegression::predict(double *input){
    double **M = new double*[B.getRows()];
    M[0] = new double(1);

    for(int i=1;i<=B.getRows();i++){
        M[i] = new double;
        M[i][0] = input[i-1];
        if(B.getRows() > data.getCols())
            if(i%2 == 0)
                M[i][0] = input[i-2]*input[i-2];
    }

    Matrix P(B.getRows(), 1, M);
    P = P.transpose()*B;

    return P[make_pair(0,0)];
}

//retorna os coeficientes encontrados
Matrix LinearRegression::getCoef(){
    return B;
}

//destrutor
LinearRegression::~LinearRegression()
{
}
