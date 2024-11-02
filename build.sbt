ThisBuild / version := "0.1.0-SNAPSHOT"


lazy val root = (project in file("."))
  .settings(
    name := "TPSparkSentiment"
  )


scalaVersion := "2.12.10"  // Match Scala version with Spark's supported versions

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.1",
  "org.apache.spark" %% "spark-sql" % "3.3.1",
  "org.apache.spark" %% "spark-mllib" % "3.5.0"
)