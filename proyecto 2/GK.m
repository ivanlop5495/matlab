%Iván Martínez López y Marcos Fraile Muñoz

%% Objetivo obligatorio, redimensionar a 500x500 o menor
im = imread("GK.png");
[filas, columnas, ~] = size(im);
if filas > columnas
    I = imresize(im, [500 NaN]);
else
    I = imresize(im, [NaN 500]);
end
%% Objetivo obligatorio, Segmentación basada en características de color, Espacio de color y análisis visual
Imagen_lab = rgb2lab(I);
a = Imagen_lab(:,:,2);
b = Imagen_lab(:,:,3);

figure;
subplot(1,2,1); mesh(a); title('Componente a');
subplot(1,2,2); mesh(b); title('Componente b');
figure;
scatter(a(:), b(:), '.'); 
xlabel('a*'); ylabel('b*'); title('Espacio de color ab');
%Identificamos clusters separados para elegir objetos.

%% Objetivo obligatorio, Segmentación basada en características de color, Selección de patches

% Fondo (hay varios fragmentos para el fondo para no coger solamente una parte
% de este y evitar así futuros errores en otros apartados, luego se
% unifican todos en un solo patch.

% Esquina superior izquierda
fondo1_a = a(10:40, 10:40); fondo1_b = b(10:40, 10:40);
% Esquina superior derecha
fondo2_a = a(10:40, 460:490); fondo2_b = b(10:40, 460:490);
% Centro lateral izquierdo
fondo3_a = a(250:280, 10:40); fondo3_b = b(250:280, 10:40);

% Concatenamos todos los fragmentos de fondo en una sola matriz de 2 columnas
patch_fondo = [ [fondo1_a(:), fondo1_b(:)]; [fondo2_a(:), fondo2_b(:)]; [fondo3_a(:), fondo3_b(:)] ];

% Pelota roja
roja_a = a(300:330, 300:330); roja_b = b(300:330, 300:330);
patch_roja = [roja_a(:), roja_b(:)];

% Pelotas verdes
verdes_a = a(350:380, 100:130); verdes_b = b(350:380, 100:130);
patch_verde = [verdes_a(:), verdes_b(:)];

% Pelota morada
morada_a = a(100:130, 230:260); m1_b = b(100:130, 230:260);
patch_morada = [morada_a(:), m1_b(:)];

%% Objetivo obligatorio, Segmentación basada en características de color, Preparación de conjuntos de observaciones balanceados

% Calculamos N píxeles totales para el balanceo 
Np0 = size(patch_fondo, 1);
Np1 = size(patch_roja, 1);
Np2 = size(patch_verde, 1);
Np3 = size(patch_morada, 1);
% Buscamos el número mínimo de píxeles entre las 4 clases para balancear
n_min = min([Np0, Np1, Np2, Np3]); 
% Mantenemos la semilla para consistencia
rng(123);
% Generamos los índices aleatorios para cada clase
ind0 = randperm(Np0, n_min);
ind1 = randperm(Np1, n_min);
ind2 = randperm(Np2, n_min);
ind3 = randperm(Np3, n_min);
%% Objetivo obligatorio, Segmentación basada en características de color, Construcción de conjuntos de entrenamiento y validación

% Creamos el vector de etiquetas balanceado para las 4 clases.
T = [zeros(n_min, 1);
     ones(n_min, 1);
     2*ones(n_min, 1); 
     3*ones(n_min, 1)];
X = [patch_fondo(ind0, :);
     patch_roja(ind1, :);
     patch_verde(ind2, :);
     patch_morada(ind3, :)];
% Definimos la partición y creamos las variables X_train, T_train, X_val y
% T_val.
cv = cvpartition(T, 'HoldOut', 0.2);
X_train = X(training(cv), :);
T_train = T(training(cv), :);
X_val   = X(test(cv), :);
T_val   = T(test(cv), :);
% Visualización mediante scatter plots para comprobar la representatividad.
figure;
subplot(1,2,1); gscatter(X_train(:,1), X_train(:,2), T_train);
title('Conjunto Entrenamiento'); xlabel('a'); ylabel('b');
subplot(1,2,2); gscatter(X_val(:,1), X_val(:,2), T_val);
title('Conjunto Validación'); xlabel('a'); ylabel('b');


%% Objetivo obligatorio, Segmentación basada en características de color, Selección del hiperparámetro k

% Probamos distintos valores de k para ver cuál ofrece mejor tasa de acierto.
valores_k = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51];
acierto_train = zeros(size(valores_k));
acierto_val = zeros(size(valores_k));
% Ajustamos el modelo k-NN.
for i = 1:length(valores_k)  
    modelo = fitcknn(X_train, T_train, 'NumNeighbors', valores_k(i));    
    % Evaluamos ahora su precisión.
    acierto_train(i) = sum(predict(modelo, X_train) == T_train) / length(T_train);
    acierto_val(i)   = sum(predict(modelo, X_val) == T_val) / length(T_val);
end

figure;
plot(valores_k, acierto_train, '-o', valores_k, acierto_val, '-s');
grid on; xlabel('k'); ylabel('Tasa de acierto');
legend('Entrenamiento', 'Validación'); title('Curva de Aprendizaje');

% Seleccionamos el mejor k y analizamos los resultados detalladamente.
[~, ind] = max(acierto_val);
k_elegido = valores_k(ind);
fprintf('El k óptimo seleccionado es: %d\n', k_elegido);
% Entrenamiento del modelo definitivo basado en color.
modelo_final = fitcknn(X_train, T_train, 'NumNeighbors', k_elegido);
% Representación de las matrices de confusión.
figure;
subplot(1,2,1); confusionchart(T_train, predict(modelo_final, X_train));
title('M. Confusión: Entrenamiento');
subplot(1,2,2); confusionchart(T_val, predict(modelo_final, X_val));
title('M. Confusión: Validación');
%% Objetivo obligatorio, Segmentación basada en características de color, Segmentación de la imagen completa

X_completa = [a(:), b(:)];
Etiquetas_I_vector = predict(modelo_final, X_completa);
Etiquetas_I = reshape(Etiquetas_I_vector, size(a));

figure;
imshow(label2rgb(Etiquetas_I)); 
title('Capa de Segmentación (Falso Color)');

%% Objetivo obligatorio, Segmentación basada en características de color, (Opcional) Filtrado posterior

Etiquetas_filtradas = medfilt2(Etiquetas_I, [9 9]); 

figure;
subplot(1,2,1); imshow(label2rgb(Etiquetas_I)); title('Segmentación Original');
subplot(1,2,2); imshow(label2rgb(Etiquetas_filtradas)); title('Segmentación con Filtro de Mediana (9x9)');
%% Objetivo obligatorio Segmentación basada en características de color y en posicionamiento

[columnas_imagen, filas_imagen] = meshgrid(1:size(I,2), 1:size(I,1));
%Fondo.
cf1 = [reshape(filas_imagen(10:40, 10:40), [], 1), reshape(columnas_imagen(10:40, 10:40), [], 1)];
cf2 = [reshape(filas_imagen(10:40, 460:490), [], 1), reshape(columnas_imagen(10:40, 460:490), [], 1)];
cf3 = [reshape(filas_imagen(250:280, 10:40), [], 1), reshape(columnas_imagen(250:280, 10:40), [], 1)];

% Unimos las coordenadas del fondo.
coordenadas_fondo = [cf1; cf2; cf3];
% Pelota roja.
coordenadas_roja = [reshape(filas_imagen(300:330, 300:330), [], 1), reshape(columnas_imagen(300:330, 300:330), [], 1)];
% Pelota verde.
coordenadas_verde = [reshape(filas_imagen(350:380, 100:130), [], 1), reshape(columnas_imagen(350:380, 100:130), [], 1)];
% peloa morada.
coordenadas_morada = [reshape(filas_imagen(100:130, 230:260), [], 1), reshape(columnas_imagen(100:130, 230:260), [], 1)];
X_4f = [patch_fondo(ind0, :),  coordenadas_fondo(ind0, :);
        patch_roja(ind1, :),   coordenadas_roja(ind1, :);
        patch_verde(ind2, :),  coordenadas_verde(ind2, :);
        patch_morada(ind3, :), coordenadas_morada(ind3, :)];

% El vector de etiquetas T_4f debe ser creado de nuevo para incluir todas
% las clases balanceadas.
T_4f = [zeros(n_min, 1); ones(n_min, 1); 2*ones(n_min, 1); 3*ones(n_min, 1)];

%% Objetivo obligatorio Segmentación basada en características de color y en posicionamiento, Normalización

X_train_4f = X_4f(training(cv), :);
T_train_4f = T(training(cv), :);
X_val_4f   = X_4f(test(cv), :);
T_val_4f   = T(test(cv), :);
mu_train = mean(X_train_4f);
sigma_train = std(X_train_4f);
X_train_norm = (X_train_4f - mu_train) ./ sigma_train;
X_val_norm   = (X_val_4f - mu_train) ./ sigma_train;
X_completa_4f = [a(:), b(:), filas_imagen(:), columnas_imagen(:)];
X_completa_norm = (X_completa_4f - mu_train) ./ sigma_train;

%% Objetivo obligatorio Segmentación basada en características de color y en posicionamiento, Repetición del proceso de diseño y evaluación

valores_k = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51];
acierto_val_4f = zeros(size(valores_k));
for i = 1:length(valores_k)
    modelo_4f = fitcknn(X_train_norm, T_train_4f, 'NumNeighbors', valores_k(i));
    acierto_val_4f(i) = sum(predict(modelo_4f, X_val_norm) == T_val_4f) / length(T_val_4f);
end
[~, ind_4f] = max(acierto_val_4f);
k_elegido_4f = valores_k(ind_4f);

% Modelo final con 4 características
modelo_final_4f = fitcknn(X_train_norm, T_train_4f, 'NumNeighbors', k_elegido_4f);
Etiquetas_4f_vec = predict(modelo_final_4f, X_completa_norm);
Etiquetas_4f = reshape(Etiquetas_4f_vec, size(a));

figure;
subplot(1,2,1); imshow(label2rgb(Etiquetas_I)); title('Solo Color (ab)');
subplot(1,2,2); imshow(label2rgb(Etiquetas_4f)); title('Color + Posición (abxy)');


%% Objetivo creativo
clear all;
clc;
close all;

I = imread('GK.png');
% Aplicamos un filtro gaussiano para reducir el ruido y el error.
I_suavizada = imfilter(I, fspecial('gaussian', 15, 4));
% Segmentamos por color.
I_hsv = rgb2hsv(I_suavizada);
H = I_hsv(:,:,1);
S = I_hsv(:,:,2);
% Capturamos balones (los colores vivos) y quitamos el fondo (verde azulado).
mask = (S > 0.25) & (H < 0.28 | H > 0.52);
mask = imfill(mask, 'holes');
mask = bwareaopen(mask, 1000);
% Empleamos la técnica de watershed.
D = -bwdist(~mask);
D = imhmin(D, 2);
L = watershed(D);
mask(L == 0) = 0;
% Quitamos errores que queden sueltos.
mask = bwareaopen(mask, 5000);
% Clasificación de los objetos.
stats = regionprops(mask, 'Area', 'Perimeter', 'Centroid', 'BoundingBox');
figure; imshow(I); hold on;
fprintf('Balones detectados: %d\n', length(stats));
for k = 1:length(stats)
   area = stats(k).Area;
   perim = stats(k).Perimeter;
   circ = (4 * pi * area) / (perim^2);
   centro = stats(k).Centroid;
   fprintf('Objeto %d -> Area: %.0f | Circularidad: %.3f\n', k, area, circ);
   % Ajustamos basandonos en el area de cada elemento (esta se puede
   % obtener mediante imtool).
  
   % La menor es la de tenis.
   if area < 50000
       tipo = 'Tenis'; col = 'y';
   % La mayor es la de balonceso.
   elseif area > 100000
       tipo = 'Basket'; col = 'r';  
   % A la hora de diferenciar la pelota de futbol y la de rugby nos fijamos
   % en la circularidad de estas.
   elseif circ < 0.70
       tipo = 'Futbol'; col = 'g';      
   % Por descarte, el que queda es el de rugby.
   else
       tipo = 'Rugby'; col = 'm';
   end
   rectangle('Position', stats(k).BoundingBox, 'EdgeColor', col, 'LineWidth', 3);
   text(centro(1), centro(2), tipo, 'Color', 'w', 'FontWeight', 'bold','BackgroundColor', 'k', 'HorizontalAlignment', 'center');
end
hold off;


