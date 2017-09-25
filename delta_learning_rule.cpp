/* @ author Shikhar Sharma
     15530 NIT Hamirpur


Implementation for Feedforward + Backpropagation network (Delta Learning Rule)
Applicable for Multi layer neural network having single o/p i.e N:N:1
Assuming learning rate(alpha) = 0.1
*/

#include <iostream>
#include <math.h>
#define e 2.7182

using namespace std;

class Neuron
{
 public: 

  double **w_; // weight of Layer 1
  double *v_; // for Hidden Layer

  double *input_, *output_; //saved for back propagation(layer 1)
  double *input__, output__; //saved for back propagation(hidden layer)
 
 public: 
  
  Neuron(int n){   
    
    input_ = new double(n);
    output_ = new double(n);
    
    v_ = new double(n+1);
    
    *w_ = new double(n+1);
    
    for(int i=0;i<n+1;i++)
      w_[i] = new double(n); 

    cout<<"\n************For Layer 1*******************\n";
    for(int i=0;i<n;i++){
      cout<<"Weights for Neuron "<<i+1<<": ";
      for(int j=0;j<n+1;j++)
        cin>>w_[j][i];
      cout<<endl;
    }
    
    cout<<"\n************For Hidden Layer**************\n";
    for(int i=0;i<n+1;i++){
      cout<<"Weight for Neuron "<<i+1<<": ";
      cin>>v_[i];
      cout<<endl;
    }  
  }

  double get_activation(double x, char ch)
  {
     //for Binary Sigmoidal Function f(x) = 1 / (1+e^-x);
     if(ch=='1'){
      double denominator = 1.0 + (pow(e,-x));   
      return 1/denominator; 
     }

     //for Bipolar Sigmoidal Function f(x) = (1-e^-x) / (1+e^-x);
     else{
      double denominator = 1.0 + (pow(e,-x));
      double numerator = 1.0 - (pow(e,-x));
      return numerator/denominator;
     }
  }

  double feed_forward(double* _input, int n, char ch)
  {
    for(int i=0;i<n;i++){
      input_[i] = _input[i];  
    }
    
    /* for multiple inputs,
     sigma = w0_ * x0_ + w1_ * x0_ ... + b*/
    
    //Calculating O/p for input to Hidden Layer
    for(int i=0;i<n;i++){
      double sigma = w_[n][i];
      for(int j=0;j<n;j++)
        sigma += w_[j][i] * _input[j]; 
      output_[i] = get_activation(sigma,ch);
    } 

    for(int i=0;i<n;i++){
      input__[i] = output_[i];  
    }

    //Calculating o/p for Hidden Layer to o/p layer
    double sigmaa = v_[n];    
    for(int i=0;i<n;i++){
      sigmaa += v_[i]*output_[i]; 
    } 

    output__ = get_activation(sigmaa,ch);
    return output__; 
  }

  double get_activation_gradient(double x, char ch)
  {
    if(ch==1){
      return (get_activation(x,ch) * (1 - get_activation(x,ch)));
    }
    else{
      return 0.5*((1 - get_activation(x,ch))*(1 + get_activation(x,ch)));
    }
  }

  void propagate_backward(double target, int n, char ch)
  {
    const double alpha = 0.25; //learning rate
    const double grad = (target - output__ ) * get_activation_gradient(output__,ch);

    for(int i=0;i<n;i++){
      v_[i] += alpha * grad * input__[i]; // last input_ came from d
      //b_[i] += alpha * grad * 1.0; // last 1.0 came from d(wx+b)/db =1  
    }

    double *zin = new double(n);
    double *grad_zin = new double(n);
    for(int i=0;i<n;i++){
      zin[i] = grad*input__[i];
      grad_zin[i] = zin[i] * get_activation_gradient(input__[i],ch); 
    }

    for(int i=0;i<n;i++){
      for(int j=0;j<n+1;j++){
        w_[j][i] += alpha * grad_zin[i] * input_[j];
      }
    }
  }

  void feed_forward_and_print(double *input,int n, char ch)
  {
    cout<<"[";
    for(int i=0;i<n;i++){
      cout << input[i] <<" ";
    }
    cout<<"] --> ";
    
    cout<<feed_forward(input,n,ch)<<"\n";
  }

};

int main()
{
  int n, target;
  cout<<"Enter the no. of Neurons in Network: ";
  cin>>n;
  cout<<endl;
  Neuron my_neuron(n);

  double *input = new double(n);  //input for each neurons

  for(int i=0;i<n;i++){
    cout<<"Input for Neuron "<<i+1<<": ";
    cin>>input[i];
  }

cout<<"\nTarget Value: ";
cin>>target;
cout<<"\n";

  char ch;
  cout<<"\nType of Activation Function: \n";
  cout<<"1) Binary Sigmoidal Function\n";
  cout<<"2) Bipolar Sigmoidal Function\n";
  cout<<"Enter your choice: ";
  cin>>ch;

  for (int r=0; r < 10; r++)
  {
    cout << "\n\nTraining " << r+1 <<endl;
    my_neuron.feed_forward_and_print(input,n,ch);
    my_neuron.propagate_backward(target, n, ch);
    
    //cout<<fixed(4)<<setprecision(3);
    for(int i=0;i<n+1;i++){
      for(int j=0;j<n;j++){
        cout<<my_neuron.w_[i][j] << " ";
      }
      cout<<"\n";  
    }

  }
  cout<<endl;
  return 0;
}
