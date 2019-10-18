#include "Matrix.h"

//Construtor padr�o
Matrix::Matrix(){}

//Construtor que recebe o n�mero de linhas N, o n�mero de colunas M e os dados da matriz
Matrix::Matrix(int N,int M, double ** data){
    R=N;
    C=M;
    this->M = new double*[N];
    for(int i=0;i<R;i++)
        this->M[i] = new double[M];

    if(data != nullptr){
        for(int i=0;i<R;i++){
            for(int j=0;j<C;j++){
                this->M[i][j] = data[i][j];
            }
        }
    }
}

//sobrecarga do operador + para realizar a soma de duas matrizes
Matrix Matrix::operator +(const Matrix M2){
    if(M2.R != R && M2.C != C)
        return Matrix(0,0);
    Matrix O(R,C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            O.M[i][j]=M[i][j]+M2.M[i][j];
        }
    }
    return O;
}

//sobrecarga do operador * para realizar a multiplica��o de duas matrizes
Matrix Matrix::operator *(const Matrix M2){
    if(M2.R!=C)
        return Matrix(0,0);
    Matrix O(R,M2.C);
    for(int i=0;i<R;i++){
        for(int k=0;k<M2.C;k++){
            int p=0;
            for(int j=0;j<C;j++){
                p+=M[i][j]*M2.M[j][k];
            }
            O.M[i][k]=p;
        }
    }
    return O;
}

//sobrecarga do operador * para multiplicar a matriz por um valor real k
Matrix Matrix::operator *(double k){
    Matrix O(R,C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            O.M[i][j]=M[i][j]*k;
        }
    }
    return O;
}

//sobrecarga do operador + para somar um valor real k a matriz
Matrix Matrix::operator +(double k){
    Matrix O(R,C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            O.M[i][j]=M[i][j]+k;
        }
    }
    return O;
}

//sobrecarga do operador << para imprimir a matriz
ostream& operator << (ostream& out, Matrix A){
    if(A.getRows()==0 || A.getCols()==0){
        printf("Matriz invalida\n");
        return out;
    }
    int L=A.getRows();
    int C=A.getCols();
    for(int i=0;i<L;i++){
        for(int j=0;j<C;j++){
            out<<A[make_pair(i,j)]<<" ";
        }
        out<<endl;
    }
    out<<endl;
    return out;
}

//sobrecarga do operador >> para ler a matriz
istream& operator >> (istream& input, Matrix &A){
    for(int i=0;i<A.getRows();i++){
        for(int j=0;j<A.getCols();j++)
            input>>A.M[i][j];
    }

    return input;
}

//sobrecarga do operador [] para acessar os dados da matriz
double Matrix::operator [] (pair<int,int> P){
    return M[P.first][P.second];
}

//m�todo para calcular o determinante de uma matriz quadrada utilizando o teorema de Laplace
double Matrix::determinante(){
    double D=0;
    if(R!=C){
        cout<<"ERRO!\n";
        return 0;
    }
    if(R==1 && C == 1){
        return M[0][0];
    }
    double cof;
    Matrix Menor(R,C);
    for(int i=0;i<C;i++){
        Menor = CofMatrix(R,C,i);

        if(i%2==1)
            cof = -1*Menor.determinante();
        else
            cof = Menor.determinante();
        D += M[0][i]*cof;
    }
    return D;
}

//M�todo para calcular a matriz de cofator x,y
Matrix Matrix::CofMatrix(int N,int M,int x, int y){
    Matrix Mn(N-1,M-1);
    int a=0,b=0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){

        //cout<<a<<" "<<b<<"\n";
            if(j!=x && i != y){
                Mn.M[b][a]=this->M[i][j];
                a++;
            }
        }
        if(a > 0)
            b++;
        a=0;
    }
    return Mn;
}

//M�todo para calcular a tranposta da matriz
Matrix Matrix::tranpose(){
    Matrix O(R,C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            O.M[i][j]=M[j][i];

        }
    }
    return O;
}

//M�todo para calcular a inversa da matriz pelo m�todo de invers�o por matriz adjunta
Matrix Matrix::inverse(){

    if(R!=C){
        cout<<"ERRO!\n";
        return Matrix();
    }
    double det = determinante();
    if(det == 0){
        cout<<"Matriz nao inversivel\n";
        return Matrix();
    }
    double cof;
    Matrix Menor(R,C);
    double **M;
    M = new double*[R];
    for(int i=0;i<R;i++){
        M[i] = new double[C];
        for(int j=0;j<C;j++){
            Menor = CofMatrix(R,C,i, j);
            if(i+j%2==1)
                cof = -1*Menor.determinante();
            else
                cof = Menor.determinante();
            M[i][j] = cof;
        }
    }
    Matrix Inv(R, C, M);
    Inv = Inv * (1.0/det);
    delete M;
    return Inv;
}

//M�todo para preencehr uma matriz com um valor real x
void Matrix::fillMatrix(double x){
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            M[i][j]=x;
        }
    }
}

//M�todo que retorna o n�mero de linhas da matriz
int Matrix::getRows(){
    return R;
}

//M�todo que retorna o n�mero de colunas da matriz
int Matrix::getCols(){
    return C;
}

//destrutor
Matrix::~Matrix()
{
    //for(int i=0;i<R;i++)
     //   delete M[i];
    //delete M;
}
