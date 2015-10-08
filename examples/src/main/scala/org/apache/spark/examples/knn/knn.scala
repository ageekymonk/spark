package org.apache.spark.examples.knn

import org.apache.spark.examples.{KNearestNeighbourModel, KNearestNeighbour}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

import scala.math._

/**
 * Created by ramz.sivagurunathan on 29/08/2015.
 */
object knn {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Pi")
    conf.set("spark.mesos.coarse", "true")
    val spark = new SparkContext(conf)

    if (args.length < 3) {
      println("Missing Arguments. Usage knn dataFileName numNeighbours numPivots")
      spark.stop()
      sys.exit()
    }

    val (dataFileName, numNeighbours, numPivots) = (args(0), args(1).toInt, args(2).toInt)

//    val parsedData = spark.textFile(dataFileName).map { line =>
//      val parts = line.split(',').map(_.toDouble)
//      LabeledPoint(parts(54), Vectors.dense(parts.slice(0, 54)))
//    }

    val parsedData = spark.textFile(dataFileName).map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(0), Vectors.dense(parts.slice(1, 28)))
    }
    val knnModel = new KNearestNeighbourModel("knn", "euclidean", parsedData, numNeighbours, numPivots)
    val fvectors = parsedData.map(_.features)
    knnModel.predict(fvectors)
//      .foreach(v => {
//      println(v._1.toString +"," + v._2.map(x => x._2).toString)
//    })
    spark.stop()
  }
}
