'''
The Hartree-Fock self-consistent field (SCF) procedure code
Sucheta Ghosh
2019/UG/037
Integrated BSMS seventh semester
Chemistry major
'''
import math
import numpy as np

# To write a dat file for storing the overlap matrix and the H_core matrix
def writedat1(mat,dat):
    for i in range(7):
        for j in range(i+1):
            dat.write(str(i+1)+"\t"+str(j+1)+"\t"+str(mat[i][j])+"\n")

# To write a dat file for storing the two electron integral            
def writedat2(mat,dat):
    for i in range(228):
            dat.write(str(mat[i])+"\n")

#To calculate the compound indices from the two electron integral index
def com(i,j):
    if i>j:
        ij = i*(i+1)/2 + j
    else:
        ij = j*(j+1)/2 + i
    return ij
        
#Reading the required files: The electron repulsion energy, The overlap matrix, The kinetic and the potential energy matrix 
repul_nrg = open("enuc.dat", 'r')
s_overlap = open("s.dat", 'r')
ke_nrg = open("ke.dat", 'r')
pe_nrg = open("pe_v.dat", 'r')
#From the kinetic and potential energy matrix, we construct the core hamiltonian and storing it.
H_core = np.zeros((7,7))
for i in ke_nrg:
    split_i = i.split()
    H_core[int(split_i[0])-1][int(split_i[1])-1]= float(split_i[2])
    if int(split_i[0]) != int(split_i[1]):
        H_core[int(split_i[1])-1][int(split_i[0])-1]= float(split_i[2])

for i in pe_nrg:
    split_i = i.split()
    H_core[int(split_i[0])-1][int(split_i[1])-1] += float(split_i[2])
    if int(split_i[0]) != int(split_i[1]):
        H_core[int(split_i[1])-1][int(split_i[0])-1] += float(split_i[2])

#Now, we will read the two electron integral terms and calculate the compound indices and hence construct the overlap matrix
#we will store the entire series of two electron integrals in a one-d array and the compound indices in another one-d array which will be our index array
two_e_term = open("two_integ.dat", 'r')
#Initializing
one_d_array = np.zeros((406,),dtype=float)
index_array = np.zeros((228,),dtype=float)
#Reading and story the two electron integral file
for i in two_e_term:
    index = i.split()
    a = int(index[0])-1
    b = int(index[1])-1
    c = int(index[2])-1
    d = int(index[3])-1
    ab = int(com(a,b))
    cd = int(com(c,d))
    abcd = int(com(ab,cd))
    one_d_array[abcd]=float(index[4])

#We are storing the overlap matrix for later use. We can print and see the overlap matrix if required.
overlap_matrix = np.zeros((7,7),dtype=float)
term_ist = np.zeros((28,),dtype=int)
term_sec = np.zeros((28,),dtype=int)
term_third = np.zeros((28,),dtype=float)

for i in s_overlap:
    s1 = i.split()
    overlap_matrix[int(s1[0])-1][int(s1[1])-1] = float(s1[2])
    if int(s1[0]) != int(s1[1]):
        overlap_matrix[int(s1[1])-1][int(s1[0])-1] = float(s1[2])
#print('The overlap matrix that we have is',overlap_matrix)   
#storing the overlap_matrix, H_core and the one d array of the two electron integral terms in seperate .dat files for later use.
writedat1(overlap_matrix,open("overlap_after.dat","w"))
writedat1(H_core,open("pe_nrg_after.dat","w"))
writedat2(one_d_array,open("one_d_array_after.dat","w"))
           
#Diagonalising the overlap matrix to calculate the S inverse half matrix.
overlap_eigval,overlap_eigenvec = np.linalg.eigh(overlap_matrix)
print("The eigenvalues of overlap matrix:\n",overlap_eigval)
overlap_diag= np.zeros((7,7))
 
for i in range(7):
    overlap_diag[i][i]= 1/(math.sqrt(overlap_eigval[i]))
    
overlap_inverse_half = np.zeros((7,7))
a= np.matmul(overlap_diag,overlap_eigenvec.transpose())
overlap_inverse_half = np.matmul(overlap_eigenvec,a)

print("The overlap_inverse_half matrix is as follows:")
for i in range(7):
    for j in range(7):
        print(overlap_inverse_half[i][j],end=" ")
    print()

#To form an initial (guess) Fock matrix in the orthonormal AO basis using the core Hamiltonian as a guess
f_init_guess= np.zeros((7,7))
a = np.matmul(H_core,overlap_inverse_half.transpose())
f_init_guess = np.matmul(overlap_inverse_half,a)

print("The initial (guess) Fock matrix:")
for i in range(7):
    for j in range(7):
        print(f_init_guess[i][j],end=" ")
    print()

#Subsequently diagonalising the Guess Fock matric to calculate the initial density matrix    
f_eigval,f_eigenvec = np.linalg.eigh(f_init_guess)    
#Transforing the eigenvector of the Fock matrix into the original (non-orthogonal) AO basis
transform_eigvec = np.matmul(overlap_inverse_half,f_eigenvec)
#We can print and check the corresponding eigenvalues and eigenvectors by uncommenting
#print("the Fock matrix eigenvalue",f_eigval)
#print("the f eigenvec",transform_eigvec)

#Now, calculating the guess density matrix
D_init= np.zeros((7,7))
for i in range(7):
    for j in range(7):
        for k in range(5):
            D_init[i,j]+=(transform_eigvec[i,k]*transform_eigvec[j,k])
            
print("The initial density matrix:\n",D_init)           

#Now calculating the initial SCF energy
e_elec = 0
for i in range(7):
    for j in range(7):
            e_elec += D_init[i,j]*(H_core[i,j]+H_core[i][j])
#print("e_elec",e_elec)

#Storing the nuclear repulsion energy term
for i in repul_nrg:
    split_i = i.split()
    e_nuc = float(split_i[0])

#Calculating the total energy for the initial SCF step
tot_nrg = e_elec + e_nuc
print('The electronic energy for the initial SCF step:',e_elec)
print('The total energy for the initial SCF step:',tot_nrg)

#Now, we are already at a position to carry our the SCF iteration to the point when the difference of energy between the nth step and (n-1)th step is less than some threshold
#Fixing the threshold to delta
delta = math.pow(10,-12)
c=0
p=1
print("Itr"+"\t\t"+"E(elec)"+" \t\t\t"+"E(tot)"+"\t\t\t\t"+"delta(E)"+"\t\t"+"RMS(D)") 
while(c!=-1): #The iterative loop
    tot_f= np.zeros((7,7))
    F= np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            for k in range(7):
                for l in range(7):
                    ijkl = int(com(int(com(i,j)),int(com(k,l))))
                    ikjl = int(com(int(com(i,k)),int(com(j,l))))
                    F[i,j] += D_init[k,l]*(2*one_d_array[ijkl]-one_d_array[ikjl])#The new Fock matrix
                    tot_f[i,j] = H_core[i,j] + F[i,j]
    #Calculating the new density by similarly diagonalising the previous step fock matrix
    f_new_guess= np.zeros((7,7))
    a1 = np.matmul(tot_f,overlap_inverse_half.transpose())
    f_new_guess = np.matmul(overlap_inverse_half,a1)     
    f_eigval_new,f_eigenvec_new = np.linalg.eigh(f_new_guess) 
    transform_eigvec_new = np.matmul(overlap_inverse_half,f_eigenvec_new)
    #print("the f_new eigenvalue",f_eigval_new)
    #print("the f_new eigenvec",transform_eigvec_new)
    D_new= np.zeros((7,7))
    rmsd = 0 #Initialising the root-mean-squared difference in consecutive densities 
    for i in range(7):
        for j in range(7):
            for k in range(5):
                D_new[i,j]+=(transform_eigvec_new[i,k]*transform_eigvec_new[j,k])#The new density matrix          
    #print("The new density matrix",D_new) 
    for i in range(7):
        for j in range(7):
            rmsd += (math.pow((D_new[i,j]-D_init[i,j]),2))    
    rmsd = math.sqrt(rmsd) #Storing the rmsd
    #Now calculating the energy for the new step     
    e_elec_new = 0
    for i in range(7):
        for j in range(7):
                e_elec_new += D_new[i,j]*(H_core[i,j]+tot_f[i][j])
    tot_nrg_new = e_elec_new + e_nuc 
            
    print(str(p)+"\t"+f'{e_elec_new:.12f}'+"\t"+f'{tot_nrg_new:.12f}'+"\t"+f'{abs(tot_nrg_new-tot_nrg):.12f}'+"\t"+f'{rmsd:.12f}')
    #If the difference in consecutive SCF energy and the root-mean-squared difference in consecutive densities fall below the delta thresholds, then we can ascertain the convergence of the procedure.
    if (abs(tot_nrg_new-tot_nrg)) <= delta and rmsd <= delta:
        c=-1
                
    else :
        tot_nrg = tot_nrg_new
        D_init = D_new
    p = p+1
 
print("Hence, the SCF converges in "+str(p)+" steps")
      







    
    
    
    