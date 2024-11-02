import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import java.io.{BufferedWriter, FileWriter}

object Train {

  def train(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("SentimentAnalysis")
      .master("local[2]")
      .getOrCreate()

    // Load dataset with headers
    val dataset = spark.read.option("header", "true").csv("src/movie.csv")
    dataset.show()

    // Filter and cast the label column to Double
    val cleanData = dataset
      .filter(col("label").isNotNull && (col("label") === "0" || col("label") === "1"))
      .withColumn("label", col("label").cast("Double")) // Ensure label is of type Double
    cleanData.show()

    println("------------------------------------")
    println(cleanData.count())
    println("------------------------------------")

    // Define stages of the pipeline
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered_words")
    val hashingTF = new HashingTF().setInputCol("filtered_words").setOutputCol("rawFeatures").setNumFeatures(10000)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    // Define models to evaluate
    val models = Seq(
      ("Logistic Regression", new LogisticRegression().setMaxIter(20)),
      ("SVM", new LinearSVC()),
      ("Decision Tree", new DecisionTreeClassifier())
    )

    val Array(trainingData, testData) = cleanData.randomSplit(Array(0.8, 0.2), seed = 1234)

    // Prepare an evaluator for model accuracy
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // Create or clear the output file
    val outputFile = "model_results.txt"
    writeToFile(outputFile, "Model Evaluation Results:\n")

    // Loop through models, train, and evaluate them
    for ((name, model) <- models) {
      // Define the pipeline with the current model
      val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, model))

      // Train the model
      val trainedModel = pipeline.fit(trainingData)

      // Make predictions
      val predictions = trainedModel.transform(testData)

      // Evaluate accuracy
      val accuracy = evaluator.evaluate(predictions)

      // Optionally, evaluate precision, recall, and F1 score
      val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
      val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
      val f1 = evaluator.setMetricName("f1").evaluate(predictions)

      // Prepare results to write to the file
      val results = s"$name: Test set accuracy = $accuracy, Precision = $precision, Recall = $recall, F1 Score = $f1"
      writeToFile(outputFile, results)

      val modelPath = s"saved_models/$name"
      trainedModel.save(modelPath)
      println(s"Model saved at: $modelPath")
    }
    spark.stop()
  }


  // Method to write results to a file
  def writeToFile(filename: String, content: String): Unit = {
    val writer = new BufferedWriter(new FileWriter(filename, true)) // true for append mode
    writer.write(content)
    writer.newLine()
    writer.close()
  }
}
