package weka.attributeSelection;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.Vector;

import org.la4j.decomposition.EigenDecompositor;
import org.la4j.matrix.sparse.CCSMatrix;

import sparce.Matrix2DSparse;

import weka.clusterers.Clusterer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;
import weka.core.neighboursearch.BallTree;

/**
 * <!-- globalinfo-start --> Performs a principal components analysis and
 * transformation of the data. Use in conjunction with a Ranker search.
 * Dimensionality reduction is accomplished by choosing enough eigenvectors to
 * account for some percentage of the variance in the original data---default
 * 0.95 (95%). Attribute noise can be filtered by transforming to the PC space,
 * eliminating some of the worst eigenvectors, and then transforming back to the
 * original space.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre> -KNN
 *  Number of neighborhood to find for KNN
 *  (default = 20)
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Isabel Cristina Pérez Verona (isabelc@informatica.unica.cu)
 * @author Tania Rama Hernández (tania.rama@yandex.com)
 * @author Jarvin Antón Vargas (janton@cav.uci.cu)
 * @author Reinier Millo Sánchez (rmillo@uclv.cu)
 */
public class LocallyLinearEmbedding2 extends UnsupervisedSubsetEvaluator implements SubsetEvaluator, OptionHandler {

  private Instances instances;

  /** Number of neighborhood to find for KNN */
  private int knn_size = 3;

  /** Embedding dimension */
//  private int dim = 3;

  /** Number of attributes */
  private int attr = 1;

  Matrix2DSparse globalWeights;
  //private Matrix inversedCorrelations;
 // private BallTree ballTree = new BallTree();

  private StringBuffer sb = new StringBuffer(2048);

  /**
   * Returns a string describing this attribute transformer
   *
   * @return a description of the evaluator suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "Poner aquí la descripción del método.";
  }

  /**
   * Returns an enumeration describing the available options.
   * <p>
   *
   * @return an enumeration of all the available options.
   **/
  public Enumeration listOptions() {
    Vector newVector = new Vector(1);

    // KNN option
    newVector.addElement(new Option("\tSet the number of neighborhood to find "
        + "for KNN.\n\t(default = 20)", "KNN", 1, "-KNN"));

   /* // DIM option
    newVector.addElement(new Option("\tSet the embedding dimension for "
        + "LocallyLinearEmbeding algorithm.\n\t(default = 3)", "DIM", 1, "-DIM"));

*/
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   * <p/>
   *
   * <!-- options-start --> Valid options are:
   * <p/>
   *
   * <pre> -KNN
   *  Number of neighborhood to find for KNN
   *  (default = 20)
   * </pre>
   *
   * <!-- options-end -->
   *
   * @param options
   *          the list of options as an array of strings
   * @throws Exception
   *           if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    resetOptions();
    String optionString;

    // KNN option
    optionString = Utils.getOption("KNN", options);
    if (optionString.length() != 0) {
      Integer temp;
      temp = Integer.valueOf(optionString);
      if(temp>1)
        setKNNSize(temp.intValue());
    }
  }

  /**
   * Reset to defaults
   */
  private void resetOptions() {
    knn_size = 20;
    attr=1;
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String KNNSizeTipText() {
    return "The maximum number of neighborhood to find for KNN.";
  }

  /**
   * Return the number of neighborhood to find for KNN
   *
   * @return number of neighborhood
   */
  public int getKNNSize() {
    return knn_size;
  }

  /**
   * Set the number of neighborhood to find for KNN
   *
   * @param knn_size
   *          number of neighborhood
   */
  public void setKNNSize(int knn_size) {
    this.knn_size = knn_size;
  }

    /**
   * Gets the current settings of LocallyLinearEmbedding
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions() {

    String[] options = new String[6];
    int current = 0;

    options[current++] = "-KNN";
    options[current++] = "" + getKNNSize();

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }

  /**
   * Returns the capabilities of this evaluator.
   *
   * @return the capabilities of this evaluator
   * @see Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    result.enable(Capability.NO_CLASS);

    return result;
  }

  Random rnd = new Random();
  @Override
  public double evaluateSubset(BitSet arg0) throws Exception {
    BitSet bs = new BitSet(arg0.length()+1);
    for(int i=0; i<arg0.length(); ++i){
        if(instances.classIndex()!=i && arg0.get(i))
            bs.set(i+1);
    }
    if(bs.cardinality()==0)
		  return 0;
    Matrix2DSparse tmpMatrix = new Matrix2DSparse(this.instances.numInstances(), this.instances.numInstances());
    calculateReconstructionWeightsMatrix(tmpMatrix,bs);
    double d1 = getMeanDifference(globalWeights,tmpMatrix);
    internalPrint(bs.cardinality()+" "+d1+" "+arg0.toString()+ " >> "+bs.toString());
    return d1;
  }

Random rnds = new Random();

private boolean isCosine = false;

  public void setCosine(boolean calc){
	  isCosine = calc;
  }

  private double getMeanDifference(Matrix2DSparse m1, Matrix2DSparse m2){
    double d1=0;

    if(isCosine){
    	m2.normalize();
    	return m1.getCosineSimilarity(m2);
    }

    m2.normalize();

    Matrix2DSparse tmp = m1.subtract(m2);
    tmp = tmp.abs();
    tmp.normalize();

    /*Matrix2DSparse tmp = m2.abs().multiply(-1);

    tmp = m1.abs().add(tmp);
    tmp = tmp.abs();
    double max = tmp.getMax();
    Matrix m;

    if(max!=0)
        tmp = tmp.divide(tmp.getMax());
    return rnds.nextInt(20); // TODO FIX it tmp.getAverageMean();*/
    return tmp.getNorm(5);
  }

  @Override
  public void buildEvaluator(Instances instances) throws Exception {
    // Store the dataset instances
    this.instances = instances;


    Iterator<Instance> itr = this.instances.iterator();
    int counter = 0;
    while(itr.hasNext()){
        Instance inst = itr.next();
        inst.setWeight(counter);
        counter++;
    }


    // Set the number of attributes
    attr = instances.numAttributes();
    globalWeights = new Matrix2DSparse(this.instances.numInstances(), this.instances.numInstances());

    BitSet bs = new BitSet();
    for(int i=1; i<=instances.numAttributes(); ++i){
    	if(instances.classIndex()!=i)
    		bs.set(i);
    }
    internalPrint("SDDSDSD "+instances.numAttributes());
    calculateReconstructionWeightsMatrix(globalWeights,bs);
    globalWeights.normalize();
  }

  private Matrix2DSparse generateEmbedding() throws Exception{
    internalPrint("MATRICES EMBBEDING");

    Matrix2DSparse sm = Matrix2DSparse.getIdentMatrix(globalWeights.getRows());

    internalPrint("OP1");
    Matrix2DSparse sm1 = globalWeights.multiply(-1);
    internalPrint("OP2");

    Matrix2DSparse sm2 = sm.add(sm1);
    internalPrint("OP3");
    Matrix2DSparse sm3 = sm2.getTranspose();
    internalPrint("OP4");


    Matrix2DSparse sm4 = sm2.multiply(sm3);
    internalPrint("OP5");

    CCSMatrix mat = new CCSMatrix(sm4.getRows(), sm4.getColumns());
    for(int i=0; i<sm4.getRows();++i){
      for(int j=0; j<sm4.getColumns();++j){
        double tmp  = sm4.getValue(i, j);
        if(tmp!=0)
          mat.set(i, j, tmp);
      }
    }
    EigenDecompositor ed = new EigenDecompositor(mat);
    internalPrint("OP6");
    org.la4j.matrix.Matrix [] xx = ed.decompose();

    internalPrint("MATRICES "+xx.length);

    for(org.la4j.matrix.Matrix m : xx){
      internalPrint("DIMS "+m.rows()+" "+m.columns());
    }


    org.la4j.matrix.Matrix D = xx[0];
    org.la4j.matrix.Matrix V = xx[1];

    List<Integer> eigenVectors = new ArrayList<Integer>();

    int vectorsCount = V.rows();
    Matrix result = new Matrix(V.columns(), attr-1);
    int index=0;
    int i = 0,j=0;
    try{
    for(j = 0; j < attr-1; j++)
    {
        double eigenValue = Double.MAX_VALUE;
        index = -1;

        for(i = 0; i < D.rows(); i++)
            if(!eigenVectors.contains(i))
                if(D.get(i, i) < eigenValue)
                {
                    eigenValue = D.get(i, i);
                    index = i;
                }
        eigenVectors.add(index);
        for(int f=0; f<V.rows(); ++f){
          result.set(index, j, V.get(index, f));
        }
    }
    }catch(Exception e){
      System.err.println(e.toString());
      System.err.println(V.rows() - 1+" "+ index);
      System.err.println(j+" x "+i);
      e.printStackTrace();
    }
    System.err.println("DIMSSS "+result.getRowDimension()+" x "+result.getColumnDimension());
    try{
    PrintWriter pw = new PrintWriter(new FileWriter("/home/millo/salida2.txt"));
    for(int x1=0; x1<result.getRowDimension();++x1){
      pw.print(result.get(x1, 0));
      for(int x2=1; x2<result.getColumnDimension();++x2){
        pw.print(","+result.get(x1, x2));
      }
      pw.println();
    }
    pw.close();
    }catch(Exception e){
      System.err.println(e.getLocalizedMessage());
      e.printStackTrace();
    }
    return null;
}



  /**
   * Generate the Inverse Correlation Matrix
   * @throws Exception
   */
  private void generateInverseCorrelationMatrix(Matrix2DSparse weights, Instance point, Instances neighbors) throws Exception {
    int icount = instances.numInstances();

    // Create a matrix with the difference between the instance and its NearestNeighbours
    Matrix result = new Matrix(neighbors.numInstances(), neighbors.numAttributes());
    int i=0;
    // Iterate over all NearestNeighbours of point
    Enumeration j =  neighbors.enumerateInstances();
    double tmp = 0;
    while(j.hasMoreElements())
    {
      Instance neighborJ = (Instance)j.nextElement();
      // Iterate all attributes
      for(int k=0; k<attr; ++k){
        tmp = neighborJ.value(k)-point.value(k);
        result.set(i, k, tmp);

      }
      ++i;
    }

    // Calculate the covariance matrix
    Matrix covariance = result.times(result.transpose());
    // Check if covariance matrix is singular
    double determinant = covariance.det();
    // Add the identity matrix if determinant is zero
    if(neighbors.size() > determinant){
      covariance.plusEquals(Matrix.identity(covariance.getRowDimension(),covariance.getColumnDimension()).times(covariance.trace()));
    }

    // Calculate the reconstruction weights W
    Matrix ones = new Matrix(neighbors.numInstances(), 1, 1.0);
    Matrix wij = covariance.solve(ones);

    // Calculate the weights sum to enforce the SUM(wij)=1
    double sum = 0;
    for(int itr = 0; itr<wij.getRowDimension(); ++itr){
      sum += wij.get(itr, 0);
    }
    // Divide the weights by the weights sum
    for(int itr = 0; itr<wij.getRowDimension(); ++itr){
      wij.set(itr, 0, (wij.get(itr, 0))/sum);
    }

    for(int itri=0; itri<wij.getRowDimension(); ++itri){
       weights.setValue((int)point.weight(), (int)neighbors.get(itri).weight(), wij.get(itri,0));
    }
  }

  private static class DistancePoint implements Comparable<DistancePoint>{
    private double distance;
    private int index;

    public DistancePoint(int index, double distance){
      this.index = index;
      this.distance = distance;
    }

    public int getIndex(){
      return index;
    }

    @Override
    public int compareTo(DistancePoint arg0) {
      if(distance<arg0.distance)
        return -1;
      return 1;
    }
  }

  /**
   * Calculate the dot product of two vectors
   * @param inst1 vector 1
   * @param inst2 vector 2
   * @return the dot product
   */
  private double dotProduct(Instance inst1, Instance inst2){
    double result = 0;
    for(int i = 0; i < attr; i++){
        result += inst1.value(i)*inst2.value(i);
    }
    return result;
}

  public String toString(){
    return sb.toString();
  }

  private Instances getKNN(Instance point, int knn_size, BitSet attr){
    String attrs = null;
    attr.clear(0);
    if(attr!=null){
        attrs = attr.toString();
        attrs = attrs.substring(1, attrs.length()-1);
        attrs = attrs.replaceAll(" ", "");
    }

    Instances result = new Instances(this.instances,knn_size);
    EuclideanDistance ed = new EuclideanDistance(result);
    if(attr!=null){
        ed.setAttributeIndices(attrs);
    }
    TreeSet<DistancePoint> tree = new TreeSet<DistancePoint>();

    for(int i=0; i<this.instances.size(); ++i){
      Instance x1 = this.instances.get(i);
      DistancePoint dp = new DistancePoint(i, ed.distance(point, x1));
      tree.add(dp);
    }

    Iterator<DistancePoint> itr = tree.iterator();
    int counter = 0;
    while(counter<knn_size && itr.hasNext()){
      DistancePoint dp = itr.next();
      counter++;
      result.add(this.instances.get(dp.getIndex()));
    }
    return result;
  }


/**
 * Calculate the best reconstructions weights for each instance and store its on a matrix
 * @throws Exception
 */
private void calculateReconstructionWeightsMatrix(Matrix2DSparse weights, BitSet attr) throws Exception{
    // Iterate over all instances
    Enumeration e = this.instances.enumerateInstances();
    while(e.hasMoreElements())
    {
        // Find K NearestNeighbours
        Instance point = (Instance)e.nextElement();
        Instances neighbors = getKNN(point, knn_size, attr);

        // Generate the inverse correlation matrix for the instance and its NearestNeighbours
        generateInverseCorrelationMatrix(weights,point,neighbors);

        //Matrix2DSparse r = generateEmbedding();
      /*  for(int i=0; i<r.getRowDimension(); ++i){
          pw.print(r.get(i, 0));
          for(int j=1; j<r.getColumnDimension(); ++j){
            pw.print(","+r.get(i, j));
          }
          pw.println();
        }
*/
        // Calculate the reconstruction weights for the instance
        //calculateReconstructionWeights(point, neighbors);
    }
}


@Override
public Clusterer getClusterer() {
  // TODO Auto-generated method stub
  return null;
}

@Override
public int getNumClusters() throws Exception {
  // TODO Auto-generated method stub
  return 0;
}

@Override
public void setClusterer(Clusterer arg0) {
  // TODO Auto-generated method stub

}

private void internalPrint(String str){
	if(false)
		System.out.println(str);
}

}
