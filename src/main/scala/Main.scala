import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object Main {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("SentimentAnalysis")
      .master("local[2]")
      .getOrCreate()

    // Example raw data to create a DataFrame
    val rawData = Seq(
      ("This movie was fantastic!", null), // null because we do not need a label for predictions
      ("I did not like the film.", null),
      ("It was an average experience.", null),
      ("this is a very unattractive movie star", null)
    )

    // Define schema for the DataFrame
    val schema = StructType(Array(
      StructField("text", StringType, true),
      StructField("label", StringType, true) // Optional for predictions
    ))

    // Create DataFrame
    val inputData = spark.createDataFrame(rawData).toDF("text", "label")

    // Show the input DataFrame
    inputData.show()

    val logisticRegressionModel = PipelineModel.load("saved_models/Logistic Regression")

    // Make predictions using the loaded model
    val predictions = logisticRegressionModel.transform(inputData)

    // Show the predictions
    predictions.select("text", "prediction").show()


  }

}
