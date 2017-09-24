/* 
  Implementation of Feedforward Network in Artificial Neural Network
*/

#include <iostream>
#include <math.h>
#define e 2.7182  

using namespace std;

class Neuron
{
 public:
  double *w_; // weight of n inputs
  double *b_; // bias

  Neuron(int n){
    w_ = new double(n);
    b_ = new double(n);

    for(int i=0;i<n;i++){
      cout<<"Weight for Neuron "<<i+1<<": ";
      cin>>w_[i];
      cout<<"Bias for Neuron "<<i+1<<": ";
      cin>>b_[i];
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

  double feed_forward(double* input, int n, char ch)
  {
    // for multiple inputs,
    // \sigma = w0_ * x0_ + w1_ * x0_ ... + b
    
    double sigma = b_[0];    
    
    for(int i=0;i<n;i++){
      sigma += w_[i]*input[i]; 
    } 
      return get_activation(sigma,ch);
  }
};

int main()
{
  int n;
  cout<<"Enter the no. of Neurons in Network: ";
  cin>>n;
  cout<<endl;
  Neuron my_neuron(n);

  double *input = new double(n);  //input for each neurons

  for(int i=0;i<n;i++){
    cout<<"Input for Neuron "<<i+1<<": ";
    cin>>input[i];
  }
  
  char ch,more='y';
  cout<<"\nType of Activation Function: \n";
  cout<<"1) Binary Sigmoidal Function\n";
  cout<<"2) Bipolar Sigmoidal Function\n";
  
while(more=='y' or more=='Y'){
  cout<<"Enter your choice: ";
  cin>>ch;
  switch(ch){
     case '1': cout<<"\nOUTPUT using Binary Sigmoidal: "<<my_neuron.feed_forward(input,n,ch)<<endl; 
              break;
     case '2': cout<<"\nOUTPUT using Bipolar Sigmoidal: "<<my_neuron.feed_forward(input,n,ch)<<endl;         
              break;
      default: cout<<"Wrong Input\n";        
  }

cout<<"\nWant to try for other Activation Function: ";
cin>>more;
cout<<endl;

}  
  return 0;
}
