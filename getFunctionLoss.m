function [loss,xi] = getFunctionLoss(svm,trainData,kertype,trainLabel,G,lambda,Fweight,C)
len=length(trainLabel);
y=svmTest_multiclass(svm,trainData,kertype,lambda);
xi=ones(1,len)-trainLabel.*(lambda*y.score+(1-lambda)*G');
xi(xi<0)=0;
loss2=C*xi*Fweight;
if(strcmp(kertype,'linear'))
    loss1=svm.w*svm.w';
else
    loss1=lambda*lambda*(svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv,kertype)*(svm.a'.*svm.Ysv)';
end
loss=loss1/2+loss2;
end