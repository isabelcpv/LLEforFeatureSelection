package weka.attributeSelection;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;

public class ProbandoMetrica extends UnsupervisedAttributeEvaluator
implements OptionHandler{

  /**
   * Returns the capabilities of this evaluator.
   *
   * @return            the capabilities of this evaluator
   * @see               Capabilities
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


  @Override
  public double evaluateAttribute(int arg0) throws Exception {
    return 0;
  }

  @Override
  public String[] getOptions() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Enumeration listOptions() {
Vector newVector = new Vector(3);

    newVector.addElement(new Option("\tCenter (rather than standardize) the" +
            "\n\tdata and compute PCA using the covariance (rather" +
            "\n\t than the correlation) matrix.",
            "C", 0, "-C"));

    newVector.addElement(new Option("\tRetain enough PC attributes to account "
                                    +"\n\tfor this proportion of variance in "
                                    +"the original data.\n"
                                    + "\t(default = 0.95)",
                                    "R",1,"-R"));

    newVector.addElement(new Option("\tTransform through the PC space and "
                                    +"\n\tback to the original space."
                                    , "O", 0, "-O"));

    newVector.addElement(new Option("\tMaximum number of attributes to include in "
                                    + "\n\ttransformed attribute names. (-1 = include all)"
                                    , "A", 1, "-A"));
    return  newVector.elements();
  }

  @Override
  public void setOptions(String[] arg0) throws Exception {
    // TODO Auto-generated method stub

  }

  @Override
  public void buildEvaluator(Instances arg0) throws Exception {
    // TODO Auto-generated method stub

  }

  public String toString() {
      return "\tPrincipal Components Attribute Transformer modificado\n\n";
  }

}
