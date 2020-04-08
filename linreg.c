#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to find the Mean of the given Array
double Mean(double arr [], int size){
    double mean = 0;
    for(int i = 0; i< size; i++){
        mean+=arr[i];
    }

    return (double)(mean/size);
}

// Function to determine the coefficients of the linear regression
double* find_a_b(double arr_x[],double arr_y[], int size){
    double *res=(double*)malloc(2*sizeof(double));
    res[1]=0;
    res[2]=0;
    double nom = 0,denom = 0;
    for(int i=0;i<size;i++){
        nom+=(arr_x[i]-Mean(arr_x,size))*(arr_y[i]-Mean(arr_y,size));
        denom += (arr_x[i]-Mean(arr_x,size))*(arr_x[i]-Mean(arr_x,size));
    }
    res[2]=nom/denom;
    res[1]=Mean(arr_y,size)-res[2]*Mean(arr_x,size);
    return res;
}

// Function to predict the wanted value
double predict (double x[], double y[], int size, double wanted){
    double *res=find_a_b(x,y,size);
    return res[1]+res[2]*wanted;
}

// Function to determine Mean Absolute Error
double MAE(double x[], double y[], int size){
    double mae = 0;
    for(int i = 0; i< size; i++){
        mae+=abs(y[i]-predict(x,y,size,y[i]));
    }
    return (double)(mae/size);
}

// Function  to determine Mean Squared Error
double MSE(double x[],double y[],int size){
    double mse = 0;
    for(int i=0; i< size; i++){
        mse+=pow((y[i]-predict(x,y,size,y[i])),2);
    }
    return (double)(mse/size);
}

//Function to determine Root Mean Square Error
double RMSE(double x[], double y[], int size){
    double rmse = 0;
    for(int i = 0; i<size;i++){
        rmse+=pow((y[i]-predict(x,y,size,y[i])),2);

    }
    rmse=rmse/size;
    return sqrt(rmse);
}

//Function to determine MEan Absolute Percentage Error
double MAPE(double x[],double y[], int size){
    double mape = 0;
    for(int i =0 ; i<size;i++){
        mape+=abs((y[i]-predict(x,y,size,y[i]))/y[i]);
    }
    return 100*(mape/size);

}

//Function to determine Mean Percentage Error
double MPE(double x[], double y[], int size){
    double mpe = 0;
    for(int i =0 ; i<size;i++){
        mpe+=(y[i]-predict(x,y,size,y[i]))/y[i];
    }
    return 100*(mpe/size);

}
int main(){
    double x[] = {1,5,7,9,14,1,20,22,26,28,30,34,36,40};
    double y[] = {3,5,9,11,15,17,21,23,27,29,33,35,37,41};
    double *a_b = find_a_b(x,y,14);
    printf("%f\t%f\n\n\n",a_b[1],a_b[2]);
    printf("%f\n\n\n",predict(x,y,14,46.88));
    printf("%f\n\n\n",MAPE(x,y,14));
    printf("%f",MPE(x,y,14));



}