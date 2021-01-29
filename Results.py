try:
  from reportlab.lib.enums import TA_JUSTIFY,TA_LEFT,TA_CENTER,TA_RIGHT
  from reportlab.lib.pagesizes import letter
  from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
  from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
  from reportlab.lib.units import inch
except:
  # import sys
  # sys.exit("Unable to find module reportlab. \nPlease install by pip install reportlab")
  raise SystemExit("Unable to find module reportlab. \nPlease install using pip install reportlab")

try:
    import pandas as pd
except:
    raise SystemExit("Unable to find module pandas.")

try:
    import seaborn as sn
except:
    raise SystemExit("Unable to find module seaborn.") 

import matplotlib.pyplot as plt

class Results:
  '''
    Class containing various methods to analyze model
  '''

  def results_initialize(self, ResultsFolderPath):
    '''
      Initializes various attributes regarding to the object.
      Args : 
        ResultsFolderPath: (string) path for creating results folder
    '''
    self.res_path = ResultsFolderPath
    try:
      os.mkdir(ResultsFolderPath)    
    except:
      pass

  def get_training_curve_for_tensorflow_model(self,model_history,EPOCHS):
    '''
      returns training curve for tensorflow model
      Args:
        model_history: Contains details regarding model training
        EPOCHS : (int) Number of epochs
    '''

    acc = model_history.history['accuracy']
    try:
      val_acc = model_history.history['val_accuracy']
    except:
      pass

    loss = model_history.history['loss']
    try:
      val_loss = model_history.history['val_loss']
    except:
      pass

    epochs_range = range(EPOCHS)
    plt.figure(num=None, figsize=(20,20), dpi=40, facecolor='w', edgecolor='k')


    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    try:
      plt.plot(epochs_range, val_acc, label = 'validation Accuracy')
    except:
      pass
    plt.legend(loc='lower right',fontsize=20)
    plt.title('Accuracy Plot',fontsize=30)
    plt.xlabel('Number of Epochs',fontsize=30)
    plt.ylabel('Accuracy',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    try:
      plt.plot(epochs_range, val_loss, label = 'validation Loss')
    except:
      pass
    plt.legend(loc='upper right',fontsize=20)
    plt.title('Loss Plot',fontsize=30)
    plt.xlabel('Number of Epochs',fontsize=30)
    plt.ylabel('Loss',fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(self.res_path+'/learning_curve.png')

  def get_confussion_matrix(self,ground_truths,predictions):
    '''
      returns condussion matrix for the model
    '''
    data = {'y_Actual': ground_truths, 'y_Predicted': predictions}

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    c_m = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(c_m, annot=True, annot_kws={"size": 16})
    plt.savefig(self.res_path+'/confussion_matrix.png')


  def generate_report(self):
    '''
      Generates report of the model
    '''
    # Creates a Report.pdf file
    doc = SimpleDocTemplate(self.res_path+"/Report.pdf",pagesize=letter,
                        rightMargin=72,leftMargin=72,
                        topMargin=72,bottomMargin=18)
    
    Story=[]
    HEADDING = "My Model"

    try:
      TRAINING_CURVE = self.res_path+"/learning_curve.png"
    except:
      raise SystemExit("Unable to find learning_curve.png")

    try:
      CONFUSSION_MATRIX = self.res_path+'/confussion_matrix.png'
    except:
      raise SystemExit("Unable to find confussion_matrix.png")

    # try:
    #   ROC_CURVE = self.res_path+"/roc_curve.png"
    # except:
    #   raise SystemExit("Unable to find roc_curve.png")


    training_plot = Image(TRAINING_CURVE, 6*inch, 6*inch)
    confussion_matrix = Image(TRAINING_CURVE, 6*inch, 6*inch)
    # roc_curve = Image(TRAINING_CURVE, 6*inch, 6*inch)

    # Heading of pdf
    styles.add(ParagraphStyle(name='Justify', alignment=TA_CENTER))
    headding = '<font size="14">%s</font>' % HEADDING
    Story.append(Paragraph(headding,styles["Heading1"]))

    try:
      # Learning curve
      styles=getSampleStyleSheet()
      styles.add(ParagraphStyle(name='Justify', alignment=TA_CENTER))
      acc_loss = '<font size="14">Accuracy And Loss Plots</font>'
      Story.append(Paragraph(headding,styles["Heading1"]))
      Story.append(training_plot)
      Story.append(Spacer(1, 12))
    except:
      pass

    try:
      # Confussion Matrix
      styles=getSampleStyleSheet()
      styles.add(ParagraphStyle(name='Justify', alignment=TA_CENTER))
      conf_mat = '<font size="14">Confussion Matrix</font>' % HEADDING
      Story.append(Paragraph(conf_mat,styles["Heading1"]))
      Story.append(confussion_matrix)
      Story.append(Spacer(1, 12))
    except:
      pass

    # try:
    #   # ROC Curves
    #   styles=getSampleStyleSheet()
    #   styles.add(ParagraphStyle(name='Justify', alignment=TA_CENTER))
    #   roc = '<font size="14">ROC Curves</font>' % HEADDING
    #   Story.append(Paragraph(roc,styles["Heading1"]))
    #   Story.append(roc_curve)
    #   Story.append(Spacer(1, 12))
    # except:
    #   pass


    # Build Report
    doc.build(Story)