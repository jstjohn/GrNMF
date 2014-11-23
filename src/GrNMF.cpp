// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;




//' GrNMF is an R/C++ implementation of Non-negative Matrix Factorization on Manifold
//' 
//' \code{GrNMF} Graphically constrained NMF
//' @importFrom Rcpp evalCpp
//' @import RcppArmadillo
//' @useDynLib GrNMF
//' @param Xr an p by n numeric matrix with each column being a feature vector, and each row being a sample vector
//' @param Wr a symmetric numeric matrix (p by p) with 1 iff xi/xj are nearest neighbors (you decide/compute this in advance, 11 is a good neighbor threshold) on a graph, and 0 otherwise.
//' @param lambda the weight to give to the Wr matrix in the NMF minimization. Lambda=0 means standard NMF (Lee's 2001 algorithm).
//' @param k the number of inner dimensions (reduced features, or 'rank') to use for the NMF algorithm
//' @param n_iter the number of optimization loop iterations.
//' @param converge threshold for convergence test for early termination. Negative values deactivate this feature. 
//' @return a list of \code{U}, 
//'   (sometimes called \code{W}) and \code{V} 
//'   (sometimes called \code{H}) for the NNLS fit.
//'   \code{Max.iter} which stores the maximum iteration before NMF converged 
//'   Additionally the standard NMF objective 
//'   function fit is returned as \code{ObjectiveFitNMF}.
//'   The full GrNMF objective function score is retured as
//'   \code{ObjectiveFitGrNMF}
//' @seealso see \code{\link{NMF}}
//' @references Cai, D., He, X., Wu, X., & Han, J. (2008). Non-negative Matrix Factorization on Manifold. 2008 Eighth IEEE International Conference on Data Mining (ICDM), 63–72. doi:10.1109/ICDM.2008.57
//' @export
//' @examples
//' # generate a synthetic dataset with known classes: 50 features, 23 samples (10+5+8)
//' \dontrun{
//' library(NMF)
//' n <- 100; counts <- c(10, 7, 3);
//' p <- sum(counts)
//' x <- syntheticNMF(n, counts)
//' dim(x)
//' # build the true cluster membership
//' groups <- unlist(mapply(rep, seq(counts), counts))
//' # run on a data.frame
//' set.seed(10)
//' system.time(res <- nmf(data.frame(x), 3, nmfAlgorithm("lee"), "random"))
//' 
//' 
//' # Now do the same for GrNMF
//' adj2=matrix(1,p,p) #dummy matrix of proper dimensions, not used since lambda=0
//' set.seed(10)
//' 
//' # turn off the graphical part for this comparison
//' # by setting lambda to 0
//' system.time(res2<-GrNMF(x,adj2,lambda=0,k=3))
//' 
//' ##
//' # now we can compare the two fits to the NMF
//' # objective function
//' 
//' # first from the NMF package
//' norm(as.matrix(data.frame(x)- (res@@fit@@W %*% res@@fit@@H)),'F')
//' # and next from GrNMF package
//' res2$ObjectiveFitNMF
//' }
// [[Rcpp::export]]
List GrNMF(NumericMatrix Xr, NumericMatrix Wr, int k=5, int lambda=100, int n_iter=5000, double converge=1e-6) {
    // n samples and p variables
    int n = Xr.nrow(), p = Xr.ncol();
    
    if(Wr.nrow() != p || Wr.ncol() != p){
        throw std::invalid_argument("The edge-weight vector W should be p by p (where p are the features, the rows of X)!");
    }
    // k must be no more than the number of features
    k = std::min(k, p);
    
    double lastFit=std::numeric_limits<double>::max();
    int convergenceCheckFrequency=50;
    
    arma::mat X(Xr.begin(), n, p, false);       // reuses memory and avoids extra copy
    arma::mat W(Wr.begin(), p, p, false);
    arma::mat U(n, k, arma::fill::randu); // initialize U,V to random values in 0-1
    arma::mat V(p, k, arma::fill::randu);
    double minX = X.min(), maxX = X.max();
    
    // set U/V to be random values in the range
    // of X, the NMF package in R recommends this
    U+=1e-4;
    U*=(maxX-minX);
    V+=1e-4;
    V*=(maxX-minX);
    
    
    // the D matrix is the diagonal matrix of row (or column since W is symmetric)
    // sums of W. 
    arma::mat D = arma::diagmat(arma::sum(W));
    arma::mat L = arma::mat(D - W);
    
    //loop over iterations
    int it=0;
    for(it = 0; it < n_iter; it++){
      
      //Eq 15 uij update:
      //Eq 15 numerator
      //arma::mat XV=X*V;
      //Eq 15 denominator
      //arma::mat UVtV = U*V.t()*V;
      // carry out Eq 15 update to U
      
      // The following does element wize multiplicative updates
      // on the element-wise division, which is equivalent
      // to the following unvectorized loop which is commented out.
      // see http://arma.sourceforge.net/docs.html#operators
      U %= (X * V) / (U * V.t() * V); // let armadillo try to further optimize this
      /*for(int i=0; i<n; i++){
        for(int j=0; j<k; j++){
          U(i,j) *= arma::as_scalar(XV(i,j) / UVtV(i,j));
        }
      }*/
      
      
      //Eq 16 vij update:
      //Eq 16 numerator
      //arma::mat XtUpLWV = X.t()*U + lambda*W*V;
      //Eq 17 denominator
      //arma::mat VUtUpLDV = V*U.t()*U+lambda*D*V;
      // carry out Eq 16 update to V
      
      V %= (X.t() * U + lambda * W * V) / (V * U.t() * U + lambda * D * V); // let armadillo try to further optimize this
      /*for(int i=0; i<p; i++){
        for(int j=0; j<k; j++){
          V(i,j) *= arma::as_scalar(XtUpLWV(i,j) / VUtUpLDV(i,j));
        }
      }*/
      
      if(it % convergenceCheckFrequency == 0){
        double fit = arma::as_scalar(arma::norm(X-U*V.t(),"fro")) + arma::as_scalar(lambda * arma::trace(V.t() * L * V));
        //allow the user to turn off by setting converge to a negative
        if(lastFit-fit <= converge && converge >= 0)
          break;
        lastFit=fit;
      }
      
      
    } //end iterations
    
    return Rcpp::List::create(
        Rcpp::Named("U") = U,
        Rcpp::Named("V") = V,
        Rcpp::Named("Max.iter") = it,
        Rcpp::Named("ObjectiveFitNMF") = arma::as_scalar(arma::norm(X-U*V.t(),"fro")),
        Rcpp::Named("ObjectiveFitGrNMF") = arma::as_scalar(arma::norm(X-U*V.t(),"fro")) + arma::as_scalar(lambda * arma::trace(V.t() * L * V))
    ) ;

}
