% Integrantes: [Nombre 1] y [Nombre 2]
clear all; close all; clc;

% 1. Lectura del vídeo
video_name = 'car.mp4';
objVideo = VideoReader(video_name);
fps_original = objVideo.FrameRate;

%% 2. Generación de videoA (Velocidad x4)
% Perfil MPEG-4 para asegurar el formato .mp4 solicitado
videoA = VideoWriter('videoA.mp4', 'MPEG-4'); 
videoA.FrameRate = fps_original;
open(videoA);

while hasFrame(objVideo)
    for i = 1:4
        if hasFrame(objVideo)
            frame = readFrame(objVideo);
            if i == 1
                writeVideo(videoA, frame);
            end
        end
    end
end
close(videoA);

%% 3. Generación de videoB (Velocidad x0.5)
objVideo.CurrentTime = 0;
videoB = VideoWriter('videoB.mp4', 'MPEG-4');
videoB.FrameRate = fps_original;
open(videoB);

while hasFrame(objVideo)
    frame = readFrame(objVideo);
    writeVideo(videoB, frame); 
    writeVideo(videoB, frame);
end
close(videoB);

%% 4. Análisis de características
pause(2); % Pausa de seguridad
archivos = {video_name, 'videoA.mp4', 'videoB.mp4'};

for i = 1:length(archivos)
    if isfile(archivos{i})
        v = VideoReader(archivos{i});
        info = dir(archivos{i});
        fprintf('\n--- %s ---\n', archivos{i});
        fprintf('Resolución: %d x %d px\n', v.Width, v.Height);
        fprintf('Frames: %d\n', v.NumFrames);
        fprintf('FPS: %.2f\n', v.FrameRate);
        fprintf('Tamaño: %d bytes\n', info.bytes);
    end
end

%% OBJETIVO CREATIVO: Restauración de vídeo (Tema 3.3)
% Comparativa entre Filtro Lineal de Promediado Temporal y
% Filtro No Lineal de Mediana 3D (Estadísticos Ordenados).

% 1. Carga del vídeo original
videoFile = 'car.mp4';
v = VideoReader(videoFile);

% Parámetros de la simulación
numFrames = 30; % Trabajaremos con un extracto corto para el análisis
framesOriginales = cell(1, numFrames);
framesRuidosos = cell(1, numFrames);

% 2. Lectura y adición de ruido artificial
% Simulamos el modelo g(n,k) = f(n,k) + w(n,k) indicado en el Tema 3.3
disp('Leyendo frames y añadiendo ruido...');
for k = 1:numFrames
    frame = readFrame(v);
    framesOriginales{k} = frame;

    % Añadimos ruido impulsivo (sal y pimienta) para simular degradación
    % Este tipo de ruido permite ver muy bien el contraste entre ambos filtros
    framesRuidosos{k} = imnoise(frame, 'salt & pepper', 0.05);
end

% Pre-asignamos memoria para los resultados
framesPromedio = cell(1, numFrames);
framesMediana = cell(1, numFrames);

% 3. Aplicación de los Filtros de Secuencias (Ventana temporal K=1)
% Analizamos una ventana de 3 frames: g(n, k-1), g(n, k), g(n, k+1)
disp('Aplicando filtros de restauración...');
for k = 2:(numFrames-1)

    % Extraemos los 3 frames de la ventana temporal
    f_prev = double(framesRuidosos{k-1});
    f_curr = double(framesRuidosos{k});
    f_next = double(framesRuidosos{k+1});

    % A. Filtro Lineal de Promediado Temporal (Tema 3.3.2.1)
    % Se asignan pesos iguales a los 3 frames: h(l) = 1/3
    f_promedio = (f_prev + f_curr + f_next) / 3;
    framesPromedio{k} = uint8(f_promedio);

    % B. Filtro de Estadísticos Ordenados: Mediana 3D (Tema 3.3.2.2)
    % Apilamos los frames en la 4ta dimensión (alto x ancho x color x tiempo)
    f_apilados = cat(4, f_prev, f_curr, f_next);
    % Calculamos la mediana a lo largo de la dimensión temporal (dimensión 4)
    f_mediana = median(f_apilados, 4);
    framesMediana{k} = uint8(f_mediana);
end

% 4. Visualización y Comparativa
% Mostramos el frame central de la secuencia para analizar los artefactos
frameAnalisis = round(numFrames / 2);

figure('Name', 'Comparativa de Restauración de Vídeo', 'Position', [100, 100, 1200, 800]);

subplot(2,2,1);
imshow(framesOriginales{frameAnalisis});
title('Frame Original (f(n,k))');

subplot(2,2,2);
imshow(framesRuidosos{frameAnalisis});
title('Frame Ruidoso (g(n,k))');

subplot(2,2,3);
imshow(framesPromedio{frameAnalisis});
title('Filtro Promediado Temporal (Lineal)');
xlabel('Nota: Observar el "ghosting" en el coche en movimiento');

subplot(2,2,4);
imshow(framesMediana{frameAnalisis});
title('Filtro Mediana 3D (No Lineal)');
xlabel('Nota: Bordes preservados, ruido eliminado');

disp('Proceso finalizado. Observa la figura generada.');