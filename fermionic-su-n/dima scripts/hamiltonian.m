% Input is z, 2x3xN array (for Li2), where N is a number of radnom configurations
% Orbitals 1-6 are occupied; orbitals 7-10 are empty

for n=1:N
for m=1:N
D(10:10)=0;
for i=1:6
D(i,i)=1;
end
for i=7:10
D(i,i)=-1;
end

% I use here the same z-amplitude for both spins. This can be easyly changed. 
for i=1:3
for j=1:2
D(i,j+6)=z(j,i,n);
D(i+3,j+8)=z(j,i,n);
D(j+6,i)=z(j,i,m);
D(j+8,i+3)=z(j,i,m);
end
end

% Caclulating overlap
S(n,m)=det(D);

% Caclulating one-electron matrix elements
fm=D^(-1)*det(D);

% Caclulating two-electron matrix elements
Fm(1:10,1:10,1:10,1:10)=0;
for i1=1:9
for i2=i1+1:10
for j1=1:9
for j2=j1+1:10
Dm=D;
Dm(i1,:)=0;
Dm(i2,:)=0;
Dm(:,j1)=0;
Dm(:,j2)=0;
Dm(i1,j1)=1;
Dm(i2,j2)=1;
Fmc=det(Dm);
if (i2 == j2 && i2 >6)
Fmc=Fmc+fm(j1,i1);
if(i1 == j1 && i1 >6)
Fmc=Fmc+det(D);
end
end
if (i2 == j1 && i2 >6)
Fmc=Fmc-fm(j2,i1);
end
if (i1 == j1 && i1 >6)
Fmc=Fmc+fm(j2,i2);
end
if (i1 == j2 && i1 >6)
Fmc=Fmc-fm(j1,i2);
end
Fm(i1,i2,j1,j2)=Fmc;
Fm(i2,i1,j2,j1)=Fmc;
Fm(i1,i2,j2,j1)=-Fmc;
Fm(i2,i1,j1,j2)=-Fmc;
end
end
end
end

% Renumbering the orfitals to correspond the order of them in "integrals" script 
in=[1 2 3 6 7 8 4 5 9 10];
for i1=1:10
for i2=1:10
for i3=1:10
for i4=1:10
Fm2(in(i1),in(i2),in(i3),in(i4))=-Fm(i1,i2,i3,i4);
end
end
end
end

% Caclulating one-electron matrix elements
for i=7:10
fm(i,i)=fm(i,i)+det(D);
end
for i1=1:10
for i2=1:10
fm1(in(i1),in(i2))=fm(i1,i2);
end
end

% Caclulating Hamiltonian; E1 and E2 are one- and two-electron parts; 1.8 is the nuclear repulsion energy.
E1=fm1.*I1;
E2=I2.*Fm2;
H(n,m)=sum(sum(E1))+sum(sum(sum(sum(E2))))/2+1.8*S(n,m);
end
end
