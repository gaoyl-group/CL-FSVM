function [S,f2] = Adam(H,alphaCoeff,alpha,Kneighbor)
%ADAM 此处显示有关此函数的摘要
%   α = 0.001,β1 = 0.9, β2 = 0.999, e = 10−8 and λ = 1 − 10−8
%alpha=0.005;
belt1=0.9;
belt2=0.999;
epsilon=1e-8;
lambda=1-(1e-8);
t=0;
belt_1t=belt1;
temp_1t=belt1;
temp_2t=belt2;
S=initialTheta(H,alphaCoeff);%H,alphaCoeff
mt=zeros(size(S));
vt=zeros(size(S));
f1=ContrastiveFunction(H,S,alphaCoeff,Kneighbor);%H,S,alpha,Kneighbor
f(1)=f1;
while t<1000
    t=t+1;
    grad=getGrad(H,S,alphaCoeff,Kneighbor);%H,S,alpha,Kneighbor
    mt=belt_1t*mt+(1-belt_1t)*grad;
    vt1=belt2*vt+(1-belt2)*(grad.^2);
    if norm(vt1)>norm(vt)
        vt=vt1;
    end
    %S=S-0.0001*grad;
    %deltaS=(alpha*sqrt(1-temp_2t)/(1-temp_1t)).*mt./(sqrt(vt)+epsilon);
    epsM=ones(size(mt))*epsilon;
    deltaS=alpha*(mt/(1-temp_1t)).*(1./(sqrt(vt/(1-temp_2t))+epsM));
    S=S-deltaS;
    f2=ContrastiveFunction(H,S,alphaCoeff,Kneighbor);
    %normdeltaS=norm(deltaS,inf);
    f(t+1)=f2;
    if abs((f1-f2)/f(1))<1e-3
        b=1;
        break;
    elseif f2>100*f(1)
        S=initialTheta(H,alphaCoeff);
        f2=f1;
        break;
    end
    f1=f2;
    belt_1t=belt1*lambda;
    temp_1t=temp_1t*belt1;
    temp_2t=temp_2t*belt2;
end
end