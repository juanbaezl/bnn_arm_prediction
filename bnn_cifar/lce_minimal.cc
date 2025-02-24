#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This file is based on the TF lite minimal example where the
// "BuiltinOpResolver" is modified to include the "Larq Compute Engine" custom
// ops. Here we read a binary model from disk and perform inference by using the
// C++ interface. See the BUILD file in this directory to see an example of
// linking "Larq Compute Engine" cutoms ops to your inference binary.

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x))                                                  \
  {                                                          \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// Tamaño de imagen CIFAR-10: 32x32x3 = 3072 bytes
constexpr int kImageSize = 32 * 32 * 3;

// Estructura para guardar una muestra: etiqueta e imagen
struct Sample
{
  int label;
  std::vector<float> image; // tamaño kImageSize
};

// Función para leer 10 muestras de un archivo binario CIFAR
std::vector<Sample> LoadCIFAR10Samples(const char *bin_filename,
                                       int num_samples = 10)
{
  std::vector<Sample> samples;
  std::ifstream file(bin_filename, std::ios::binary);
  if (!file)
  {
    fprintf(stderr, "No se pudo abrir el archivo: %s\n", bin_filename);
    exit(1);
  }
  samples.reserve(num_samples);

  for (int i = 0; i < num_samples; ++i)
  {
    Sample sample;

    // Leer la etiqueta (1 byte) y convertirla a int
    unsigned char label;
    file.read(reinterpret_cast<char *>(&label), 1);
    sample.label = static_cast<int>(label);
    if (file.eof())
      break;

    // Leer la imagen (3072 bytes) en un buffer temporal
    std::vector<unsigned char> buffer(kImageSize);
    file.read(reinterpret_cast<char *>(buffer.data()), kImageSize);

    // Convertir cada valor a float y normalizar dividiendo por 255.0
    sample.image.resize(kImageSize);
    for (int j = 0; j < kImageSize; ++j)
    {
      sample.image[j] = static_cast<float>(buffer[j]) / 255.0f;
    }

    samples.push_back(sample);
  }

  file.close();

  // Ejemplo: Imprimir la etiqueta y el valor normalizado del primer píxel de la
  // primera imagen
  if (!samples.empty())
  {
    std::cout << "Primera imagen, etiqueta: " << samples[0].label
              << ", primer pixel: " << samples[0].image[0] << std::endl;
  }
  return samples;
}

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    fprintf(stderr, "Uso: %s <tflite_model> <cifar_bin> <n_muestras>\n",
            argv[0]);
    return 1;
  }
  const char *filename = argv[1];
  const char *bin_filename = argv[2];
  int num_images = std::atoi(argv[3]);

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  std::vector<Sample> samples = LoadCIFAR10Samples(bin_filename, num_images);
  if (samples.empty())
  {
    fprintf(stderr, "No se pudieron leer muestras del archivo.\n");
    return 1;
  }
  printf("Se han cargado %zu muestras.\n", samples.size());

  std::vector<int> predictions;
  predictions.reserve(samples.size());
  int correct = 0;
  // Medir el tiempo de inferencia para las 10 imágenes
  for (size_t i = 0; i < samples.size(); i++)
  {
    const Sample &sample = samples[i];

    float *input_data = interpreter->typed_tensor<float>(0);
    std::copy(sample.image.begin(), sample.image.end(), input_data);

    // Ejecutar la inferencia
    auto start = std::chrono::high_resolution_clock::now();
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    auto end = std::chrono::high_resolution_clock::now();

    // Leer la salida:
    float *output_data = interpreter->typed_output_tensor<float>(0);

    // Se puede hacer un argmax de la salida para conocer la predicción
    int predicted_class = std::distance(
        output_data, std::max_element(output_data, output_data + 10));

    predictions.push_back(predicted_class);

    if (predicted_class == sample.label)
    {
      correct++;
    }
    std::chrono::duration<double> elapsed = end - start;
    printf("Imagen %zu: Etiqueta real = %d, Predicción = %d, tiempo: %fs\n", i,
           sample.label, predicted_class, elapsed.count());
  }
  printf("Predicciones correctas: %d de %zu\n", correct, samples.size());
  printf("Precisión: %.2f%%\n", 100.0 * correct / samples.size());
  return 0;
}