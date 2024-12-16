function [trainTime,svm] = trainFSVM(trainData1,trainLabel1,trainData2,trainLabel2,Fweight,kertype,C,G,paraLambda)
trainData = [trainData1,trainData2];trainLabel = [trainLabel1,trainLabel2];
tic;
options=optimset;
options.LargerScale='off';
options.Display='off';
n=length(trainLabel);
%quadprog is 1/2*x^T*H*x
H=paraLambda*paraLambda*(trainLabel'*trainLabel).*kernel(trainData,trainData,kertype);
%Gi has been changed by lambda
f=-ones(n,1)+(1-paraLambda)*G.*trainLabel';
A=[];
b=[];
Aeq=trainLabel;
beq=0;
lb=zeros(n,1);
ub=C*Fweight;
a0=zeros(n,1);
[a,fval,eXitflag,output,lambda]=quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
[svm,sv_label] = calculate_rho(a,trainData',trainLabel',C,kertype,paraLambda);
trainTime = toc;
