 //  script pour afficher le nom du fichier sélectionné
 const fileInput = document.getElementById('file-upload-button');
 const fileNameDisplay = document.getElementById('file-name');
 
 fileInput.addEventListener('change', () => {
     if (fileInput.files.length > 0) {
         fileNameDisplay.textContent = `Fichier sélectionné : ${fileInput.files[0].name}`;
     } else {
         fileNameDisplay.textContent = 'Aucun fichier sélectionné';
     }
 });