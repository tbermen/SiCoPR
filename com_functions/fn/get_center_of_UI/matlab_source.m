function half_UI=get_center_of_UI(samples_per_UI)
%half_UI reveals which value to use for the center of the UI.  For eye
%width calculations, it is necessary to place the cursor in the center of the
%UI window to ensure a 0 crossing on both left/right inside the window.
%This function was written in order to support even and odd samples_per_UI
%and to prevent the ambiguity of using samples_per_UI/2 vs. samples_per_UI/2+1

%The UI window goes from 0 to 1 with 1/samples_per_UI steps
UI_window=0:1/samples_per_UI:1-1/samples_per_UI;
%the center of the UI is sample closest to 0.5
[temp_diff,half_UI]=min(abs(UI_window-0.5));
