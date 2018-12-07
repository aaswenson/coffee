% UWNFP Code
% Authors: Marissa Brown, Dan Cech, & Alex Swenson
% NEEP 423
% Last Update: 10/19/2016

% This script uses a finite difference approximation for determining heat
% transfer in a PWR fuel pin.

clear;clc;
%% INPUTS
n0 = 1e10;
lambda = 0.4;
beta = 0.0065;
T0 = 500;
l = 2e-4;
%% Numerical Solution
N=10000;
time = 100;
tme = zeros(N,1);
for i=1:N;
    tme(i) = time*(i-1)/(N-1);
end
dtime = time/N;
k = 1e-10;
Temp = zeros(N,1);
n = zeros(N,1);
c = zeros(N,1);
n(1) = n0;
c(1) = beta*n0/(lambda*l);
Temp(1) = T0;
rho = 6.7e-4; %     $0.10
for i=2:N;
    c(i) = dtime*(beta/l*n(i-1) - lambda*c(i-1)) + c(i-1);
    n(i) = n(i-1) + dtime*(((rho-beta)/l)*n(i-1)+lambda*c(i-1));
    Temp(i) = Temp(i-1) + dtime*k*n(i-1);
end
aaa =1;
figure(1)
plot(tme,n)
figure(2)
plot(tme,Temp)
% while diff > converge;
%     N=N+1;
dr_f= R_f/N;
Tf = zeros(N,1);
Tg = Tf;
Tc = Tf;
rf=zeros(N,1);
T_s =630;
b = zeros(N,1);
a = zeros(N,N);

% Radius 
for i=1:N;
    rf(i) = R_f*(i-1)/(N-1);
end
% Centerline Temperature Boundary Condtion
% dT/dr = 0  @ r=0
for i=1;
    b(i)=0;
    a(i,i) = 1;
    a(i,i+1) = -1;
end

for i=2:N-1;
    a(i,i-1) = rf(i)/dr_f;
    a(i,i) = -(1 + 2*rf(i)/dr_f);
    a(i,i+1) = 1+rf(i)/dr_f;

    b(i)=-q_gen*rf(i)*dr_f/k_f;
end
a(N,N)=1;
b(N)=T_s;

%% solution

% Centerline A-Matrix coefficient reassignment

a(1,2)=a(1,2)/a(1,1);
b(1)=b(1)/a(1,1);

for i=2:N-1
    a(i,i+1) = a(i,i+1)/(a(i,i)-a(i,i-1)*a(i-1,i));
    b(i)= (b(i)-a(i,i-1)*b(i-1))/(a(i,i)-a(i,i-1)*a(i-1,i));
end 

b(N)=(b(N)-a(N,N-1)*b(N-1))/(a(N,N)-a(N,N-1)*a(N-1,N));
Tf(N)=b(N);
for i=(N-1):(-1):1;
Tf(i) = b(i)-a(i,i+1)*Tf(i+1);
end

% diff = abs(Tf(1)-T_old);
% T_old=Tf(1);

figure(1)
plot(rf,Tf)
hold on


