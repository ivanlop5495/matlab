% Integrantes: Iván Martínez López y Marcos Fraile Muñoz

%% Segmentación de objetos en la imagen (0.5p)

I_original = imread('GK.png'); 

% Convertimos al espacio de color HSV
I_hsv = rgb2hsv(I_original);
H = I_hsv(:,:,1); 
S = I_hsv(:,:,2);
V = I_hsv(:,:,3);

% El fondo verde tiene un rango de tono aproximado entre 0.35 y 0.55, por
% lo que creamos una máscara para este y luego la invertimos para quedarnos con los objetos.
mascara_fondo_verde = (H > 0.35) & (H < 0.58) & (S > 0.2);
I_segm = ~mascara_fondo_verde;

% Definimos los bordes como fondo
I_segm(1:15, :) = 0; I_segm(end-15:end, :) = 0;
I_segm(:, 1:15) = 0; I_segm(:, end-15:end) = 0;

figure; imshow(I_segm); title('imagen segmentada binaria');

%% Preparación de la imagen (1.5p)
% Se rellenan los huecos
I_rellena = imfill(I_segm, 'holes');
I_limpia = imopen(I_rellena, strel('disk', 5));

% Calculamos la transformada de distancia inversa
D = bwdist(~I_limpia);

% Se suaviza la matriz usando imgaussfilt
D = imgaussfilt(D, 2); 

% Hacemos que los centros sean mínimos en la matriz
D = -D;
D(~I_limpia) = Inf;

% Aplicamos watershed para separar los objetos mejor
L = watershed(D);
I_3obj = I_limpia;
I_3obj(L == 0) = 0; 
I_3obj = imerode(I_3obj, strel('disk', 7));

figure; imshow(I_3obj); title('imagen con objetos ya separados visualmente');
%% Segmentación de un único objeto mediante un único operador morfológico (1.5p)

% Extraemos el objeto con mayor area de los definidos
I_objeto_unico = bwpropfilt(I_3obj, 'Area', 1, 'largest');
I_1obj = I_original;
R = I_1obj(:,:,1); G = I_1obj(:,:,2); B = I_1obj(:,:,3);

% Cambiamos los píxeles a magenta únicamente en la zona de la pelota elegida
R(I_objeto_unico) = 255;
G(I_objeto_unico) = 0;
B(I_objeto_unico) = 255;

I_1obj = cat(3, R, G, B);

figure; imshow(I_1obj); title('Segmentación de un único objeto');
%% Segmentación basada en hit-miss (2p)

% Extraemos la plantilla ahora
propiedades = regionprops(I_objeto_unico, 'Image');
B_fg = propiedades.Image; 

% Ahora para el fondo
se_marco = strel('disk', 5);
B_bg = imdilate(B_fg, se_marco) - B_fg; 

% Y ahora implementamos el hit-miss
I_hit = imerode(I_3obj, B_fg); 
I_miss = imerode(~I_3obj, B_bg);

% Sacamos el pixel central con una intersección
I_hm_puntos = I_hit & I_miss;

% Situamos una cruz visible en este centro
I_hm_visualizar = imdilate(I_hm_puntos, strel('line', 35, 0)) | imdilate(I_hm_puntos, strel('line', 35, 90));

I_hitmiss = I_original;
R_hm = I_hitmiss(:,:,1); G_hm = I_hitmiss(:,:,2); B_hm = I_hitmiss(:,:,3);

% Definimos un color para esta
R_hm(I_hm_visualizar) = 0;
G_hm(I_hm_visualizar) = 255;
B_hm(I_hm_visualizar) = 255;

I_hitmiss = cat(3, R_hm, G_hm, B_hm);

% Representación de las imagenes ahora
figure('Name', 'Comparativa final de segmentaciones');
subplot(1, 2, 1);
imshow(I_1obj);
title('Segmentación por región');

subplot(1, 2, 2);
imshow(I_hitmiss);
title('Localización por plantilla');

%% Objetivo creativo (2 puntos)

% En este objetivo creativo buscamos mediante un filtrado por componentes
% conexas, de esta forma podemos filtrar por el área del objeto y de forma
% similar a la entrega anterior, filtrar el objeto que queramos por su área, en este caso es la pelota de tenis

I_mascara = I_3obj; 

% Usamos bwpropfilt para quedarnos con las 3 componentes de mayor tamaño
% de la escena. Por lo que eliminaríamos la pelota de tenis que es la que
% queremos filtrar sin importar qué radio de disco usemos ni cuánto se hayan encogido antes.

I_solo_balones = bwpropfilt(I_mascara, 'Area', 3, 'largest');

% Mediante una resta lógica, lo que pertenece a la máscara original pero no a los balones grandes, es la pelota de tenis.
I_solo_tenis = I_mascara & ~I_solo_balones; 

% Copiamos la imagen original para pintarla encima
I_creativa = I_original;
R_c = I_creativa(:,:,1); G_c = I_creativa(:,:,2); B_c = I_creativa(:,:,3);

% 1. Pintamos los 3 balones grandes en amarillo [255, 255, 0]
R_c(I_solo_balones) = 255; 
G_c(I_solo_balones) = 255; 
B_c(I_solo_balones) = 0;

% 2. Pintamos la pelota de tenis en rojo [255, 0, 0]
R_c(I_solo_tenis) = 255; 
G_c(I_solo_tenis) = 0; 
B_c(I_solo_tenis) = 0;

I_creativa = cat(3, R_c, G_c, B_c);
figure('Name', 'Clasificación por escala');
subplot(1, 3, 1); imshow(I_solo_balones); 
title('Balones grandes aislados');

subplot(1, 3, 2); imshow(I_solo_tenis); 
title('Pelota de tenis aislada');

subplot(1, 3, 3); imshow(I_creativa); 
title('Clasificación final');