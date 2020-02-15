#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <armadillo>
#include <utility>
#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/random.hpp>

typedef std::vector< double > state_type;

class harm_osc 
{

    std::vector< double >& m_G;
    std::vector< double >& m_B;
    int m_N;
    std::vector< double >& m_M;
    arma::Mat<double> &m_K;
    std::vector< double >& m_L;
    std::vector< double >& m_F;
    std::vector< double >& m_W;

	public:
    harm_osc( std::vector< double > &G,std::vector< double > &B, int N, std::vector< double > &M,arma::Mat<double> &K,std::vector< double > &L,std::vector< double > &F,std::vector< double > &W ) : m_G(G) , m_B(B) , m_N(N), m_M(M) , m_K(K) , m_L(L), m_F(F), m_W(W) { }

    void operator() ( const state_type &x , state_type &dxdt , const double t  )
    {
    	double sum=0;
        for (int i = 0; i < m_N; ++i)
        {
        	sum=0;
        	for (int j = 0; j < m_N; ++j)
    		{
    			sum=sum+m_K(i,j)*(x[2*i]-x[2*j]);
    		}
    		sum=sum/m_M[i];


    	
    		if(t<=20000)
    		{
        		dxdt[2*i]=x[2*i+1];
        		dxdt[2*i+1]= -(m_G[i]/m_M[i])*x[2*i+1]-(m_K(i,i)/m_M[i])*(x[2*i])-(m_B[i]/m_M[i])*pow(x[2*i],3)+m_F[i]*cos(m_W[i]*t)-sum;
    		}
    		else
    		{
        		dxdt[2*i]=x[2*i+1];
        		dxdt[2*i+1]= -(m_G[i]/m_M[i])*x[2*i+1]-(m_K(i,i)/m_M[i])*(x[2*i])-(m_B[i]/m_M[i])*pow(x[2*i],3)-sum;
    		}

        }
    }
};

struct push_back_state_and_time
{
    std::vector< state_type >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< state_type > &states , std::vector< double > &times ) : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x );
        m_times.push_back( t );
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void inicialcond(state_type &x,int N,boost::mt19937 &rng,double Ampin, double Velin)
{
    	FILE *w= fopen("Xi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%f  ",Ampin );
			fprintf(w, "%f\n",Velin );
		}
		fclose(w);
		FILE *r= fopen("Xi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &x[2*i]); // posicion inicial i
			fscanf(r, "%lf", &x[2*i+1]); // momento inicial i
		}
		fclose(r);


}

void fillK(arma::Mat<double> &K,int N,boost::mt19937 &rng,int caso,double k)
{
    boost::uniform_real<> unif( 0, 5 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if (caso==0)
    {
    	FILE *w= fopen("Ki.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				if(i==j)
				{
					fprintf(w, "%f  ",1.0);
				}
				else
				{
					fprintf(w, "%f  ",k);
				}

			}
		}
		fclose(w);
		FILE *r= fopen("Ki.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				fscanf(r,"%lf",&K(i,j));
			}
		}
		fclose(r);
    	for (int i = 0; i < N; ++i)
		{
			for (int j = N-1; j > i; --j)
			{
				K(i,j)=K(j,i);
			}
		}
    }
}

void fillM(std::vector<double> &M,int N,boost::mt19937 &rng,int caso)
{
    if (caso==0)
    {
    	FILE *w= fopen("Mi.txt", "w");
		for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 1.0);
		}
		fclose(w);
		FILE *r= fopen("Mi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &M[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
    	printf("loading Mi\n");
		FILE *r= fopen("Mi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &M[i]);
		}
		fclose(r);
    }

}

void fillG(std::vector<double> &G,int N,boost::mt19937 &rng,int caso)
{

    boost::uniform_real<> unif( 0.05, 1.0 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Gi.txt", "w");
		for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 0.1);
		}
		fclose(w);
		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
	}
	if(caso==1)
	{
    	printf("loading Gi\n");
		FILE *r= fopen("Gi.txt", "r");
		for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &G[i]);
		}
		fclose(r);
	}
}

void fillB(std::vector<double> &B,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 1 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Bi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 0.1*(4.0/3.0));
		}
		fclose(w);
		FILE *r= fopen("Bi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &B[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
    	printf("loading Bi\n");
		FILE *r= fopen("Bi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &B[i]);
		}
		fclose(r);
    }

}

void fillL(std::vector<double> &L,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( -10, 10 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Li.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", 0.0);
		}
		fclose(w);
		FILE *r= fopen("Li.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &L[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
    	printf("loading Li\n");
		FILE *r= fopen("Li.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &L[i]);
		}
		fclose(r);
    }
}

void fillF(std::vector<double> &F,int N,boost::mt19937 &rng,int caso)
{
    boost::uniform_real<> unif( 0, 10 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Fi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
				fprintf(w, "%lf  ", 1.0);
		}
		fclose(w);
		FILE *r= fopen("Fi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &F[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
    	printf("loading Fi\n");
		FILE *r= fopen("Fi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &F[i]);
		}
		fclose(r);
    }
}

void fillFw(std::vector<double> &Fw,int N,boost::mt19937 &rng,int caso,double Win)
{
    boost::uniform_real<> unif( 0, 10 );//la distribucion de probabilidad uniforme entre cero y 2pi
    boost::variate_generator< boost::mt19937&, boost::uniform_real<> > gen( rng , unif );//gen es una funcion que toma el engine y la distribucion y devuelve el numero random

    if(caso==0)
    {
    	FILE *w= fopen("Fwi.txt", "w");
    	for (int i = 0; i < N; ++i)
		{
			fprintf(w, "%lf  ", Win);
		}
		fclose(w);
		FILE *r= fopen("Fwi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
    if(caso==1)
    {
    	printf("loading Fwi\n");
		FILE *r= fopen("Fwi.txt", "r");
    	for (int i = 0; i < N; ++i)
		{
			fscanf(r, "%lf", &Fw[i]);
		}
		fclose(r);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*void printsave(size_t steps, std::vector< state_type > &x_vec,std::vector<double> &times,int N) //1 tiempo. 2 posicion. 3 momento. 4 energia potencial. 5 energia cinetica. 6 energia. 7 energia Total
{

	FILE *f1=fopen("save_1.txt","w");
	FILE *f2=fopen("save_2.txt","w");
	FILE *f3=fopen("save_3.txt","w");
	FILE *f4=fopen("save_4.txt","w");

	#pragma omp parallel
	#pragma omp for
	for( size_t i=0; i<steps; ++i )
	{
		if(i>=0 && i<steps/4)
		{
			if(i%(steps/400)==0 && i < steps/4)
			{
				printf("printing savestate: %d \n", (int)(400.0*i/steps));
			}
			fprintf(f1,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f1,"%lf	 %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f1,"\n");
		}
		if(i>=steps/4 && i<steps*2/4)
		{
			fprintf(f2,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f2,"%lf	 %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f2,"\n");
		}
		if(i>=steps*2/4 && i<steps*3/4)
		{	
			fprintf(f3,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f3,"%lf	 %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f3,"\n");
		}
		if(i>=steps*3/4)
		{
			fprintf(f4,"%lf  ",times[i] );
			for (int j = 0; j < N; ++j)
			{
				fprintf(f4,"%lf	 %lf   ",x_vec[i][2*j],x_vec[i][2*j+1]); //1 posicion. 2 momento. 3 energia potencial. 4 energia cinetica. 5 energia total
			}
			fprintf(f4,"\n");
		}
	}	
	fclose(f1);
	fclose(f2);
	fclose(f3);
	fclose(f4);
}*/

double calcampi(size_t steps,std::vector< state_type > &x_vec,int j)
{
	double max=0;
	double min=0;
	for (int i = (int)steps*9.0/10.0; i < (int)steps; ++i)
	{
		if(max<=x_vec[i][2*j])
		{
			max=x_vec[i][2*j];
		}
		if(min>=x_vec[i][2*j])
		{
			min=x_vec[i][2*j];
		}
	}
	return((max-min)/2.0);
}

void calcamp(size_t steps,std::vector< state_type > &x_vec, int N, double &Amax)
{
	std::vector<double> Amp(N);

	for (int i = 0; i < N; ++i)
	{
		Amp[i]=calcampi(steps,x_vec,i);
	}
	double max=0;
	for (int i = 0; i < N; ++i)
	{
		if(max<=Amp[i])
		{
			max=Amp[i];
		}
	}
	Amax=max;
}


void printAmp(double k, double r, double Amax, double Amin)
{
    	FILE *w= fopen("Amp.txt", "a");
		fprintf(w, "%lf   %lf   %lf   %lf\n",k,r,Amax,Amin );

		fclose(w);
}

double iterateK(int N,boost::mt19937 &rng,double Ampin, double Velin,double Win)
{
    using namespace std;
    using namespace boost::numeric::odeint;
    arma::Mat<double> K(N,N);
    std::vector<double> M(N);
    std::vector<double> G(N);
    std::vector<double> B(N);
    std::vector<double> L(N);
    std::vector<double> F(N);
    std::vector<double> W(N);
	state_type x(2*N); //condiciones iniciales
    std::vector<state_type> x_vec;
    std::vector<double> times;
    double Amax;
    ////////////////////////////////////////////////////////////////////
	fillK(K,N,rng,0,1.0);
	fillM(M,N,rng,0);
	fillG(G,N,rng,0);
	fillB(B,N,rng,0);
	fillL(L,N,rng,0);
	fillF(F,N,rng,0);
	fillFw(W,N,rng,0,Win);

	inicialcond(x,N,rng,Ampin,Velin);
	
	////////////////////////////////////////////////////////////////////
    harm_osc ho(G,B,N,M,K,L,F,W);
    runge_kutta4 < state_type > stepper;
	size_t steps = integrate_adaptive(stepper, ho, x , 0.0 , 50*2.0*M_PI/Win , 0.01, push_back_state_and_time( x_vec , times )); //1 funcion. 2 condiciones iniciales. 3 tiempo inicial. 4 tiempo final. 5 dt inicial. 6 vector de posicion y tiempo
	calcamp(steps,x_vec,N,Amax);
	return(Amax);
}


int main()
{
    using namespace std;
    using namespace boost::numeric::odeint;

    boost::mt19937 rng(static_cast<unsigned int>(std::time(0))); 

///////////////////////////////////////////////////////////////////////
    int NA;
    printf("NA: ");
    std::cin >>NA; 

    int NV;
    printf("NV: ");
    std::cin >>NV; 

    double Amod;
    printf("Amod: ");
    std::cin >>Amod; 

    double Win;
    printf("W: ");
    std::cin >>Win; 

    double Vmod;
    Vmod=Amod*Win;

////////////////////////////////////////////////////////////////////////////
	char buf[0x100];
	snprintf(buf, sizeof(buf), "PS_w%.2f.txt", Win);
    FILE *w;
    w= fopen(buf, "w");

    for (int i = 0; i < NA; ++i)
    {
    	printf("porcentaje: %lf \n", (double)100.0*(i+1)/NA);
    	for (int j = 0; j < NV; ++j)
    	{
    		fprintf(w, "%lf   %lf   %lf\n",(double)(-Amod+i*2.0*Amod/(NA-1.0)),(double)(-Vmod+j*2.0*Vmod/(NV-1)),iterateK(1,rng,-Amod+i*2.0*Amod/(NA-1),-Vmod+j*2.0*Vmod/(NV-1),Win));
		}
    }
    fclose(w);
    return 0;

}