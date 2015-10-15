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

    if (args.length < 5) {
      println("Missing Arguments. Usage knn dataFileName numNeighbours numPivots numDimensions")
      spark.stop()
      sys.exit()
    }

    val (dataFileName, numNeighbours, numPivots, numDimensions, outfile) = (args(0), args(1).toInt, args(2).toInt, args(3).toInt, args(4))
    println(f"Calculating for $dataFileName%s(D=$numDimensions%d) with k=$numNeighbours%d Pivots=$numPivots%d")
//    val parsedData = spark.textFile(dataFileName).map { line =>
//      val parts = line.split(',').map(_.toDouble)
//      LabeledPoint(parts(54), Vectors.dense(parts.slice(0, 54)))
//    }

    val parsedData = spark.textFile(dataFileName).map { line =>
      val parts = line.split(',').map(_.toDouble.abs)
      LabeledPoint(parts(0), Vectors.dense(parts.slice(1, 1 + numDimensions)))
    }
    val knnModel = new KNearestNeighbourModel("knn", "euclidean", parsedData, numNeighbours, numPivots)
    val fvectors = parsedData.map(_.features)
    knnModel.predict(fvectors, outfile)
//      .foreach(v => {
//      println(v._1.toString +"," + v._2.map(x => x._2).toString)
//    })
    spark.stop()
  }
}
