function result = svmTest_multiclass(svm, testData, kertype,paraLambda)   
if nargin < 4
    paraLambda=1;
end

if(strcmp(kertype,'linear'))
    result.score = svm.w*testData + svm.b;
else
    w = paraLambda*(svm.a'.*svm.Ysv)*kernel(svm.Xsv,testData,kertype);
    result.score = w + svm.b;
end
end