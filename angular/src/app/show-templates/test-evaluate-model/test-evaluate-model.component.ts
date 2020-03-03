import { Component, OnInit ,Input} from '@angular/core';
import { ApiService } from '../../api-service/api.service';
import { SelectTrainModelComponent } from '../select-train-model/select-train-model.component';
@Component({
  selector: 'app-test-evaluate-model',
  templateUrl: './test-evaluate-model.component.html',
  styleUrls: ['./test-evaluate-model.component.css'],
  providers: [SelectTrainModelComponent]
})
export class TestEvaluateModelComponent implements OnInit {
  @Input() header:string;
  templates: String[];
  selectedTemplate: string;
  showResult = false;
  document: any;
  fileToUpload: File = null;
  resultData: any;
  fileName: any;
  files;
  extension;
  objKeys = Object.keys;
  imageAlt = '';
  imageLoading = false;
  tables = {};
  showTable=false;
  column_length = 0;
  selectedImage:string;
  debugOutput: any;
  testModel;
  evaluateModel;
  urls;
  constructor(private apiservice: ApiService,private selectTrain: SelectTrainModelComponent) { }

  ngOnInit() {
    console.log(this.header)
    console.log(this.selectTrain.globalStr)
    this.selectTrain.globalStr="changed text"
    if(this.header=="chooseTest"){
      this.testModel=true;
    }
    else if(this.header=="chooseEvaluate"){
      this.evaluateModel=true;     
  }
  }
  readURL(files:File){
    // if($('#sampleImageDropdown').val() != 'Select Sample Image'){
    //   $('#sampleImageDropdown').val('Select Sample Image')
    // }
      console.log(files,"files uploaded",Object.keys(files).length)
      const formData: FormData = new FormData();
      this.fileName={};
      for(let fileLength=0 ; fileLength<Object.keys(files).length;fileLength++){
          formData.append('file'+fileLength.toString(), files[fileLength], files[fileLength].name);
          // this.files = files;
          
          this.fileName[fileLength] = files[fileLength].name
        }
      
      // this.extension = this.fileName.split(".")[1]
      
      console.log(formData,"______________________________");
      // this.changeFile=false
      ////


      this.apiservice.sendImages(formData).subscribe(
        resp => {
          
          console.log("result from backend :",resp);
          let imageFile = new FileReader()
          imageFile.readAsBinaryString(resp["imageList"])
          
          
        },
        err =>{
          console.log(err);
        }
        
      );
      


    // if (files) {
    //   for(let fileindex=0 ; fileindex<Object.keys(files).length;fileindex++){
    //       if(files[fileindex]){
    //         const file = files[fileindex];
    //         this.fileToUpload = files[fileindex];
    //         this.fileName = this.fileToUpload.name;
    //         const reader = new FileReader();
    //         reader.onload = e => (this.document = reader.result);
    //         console.log(reader.result)
    //         reader.readAsDataURL(file);
    //         this.showResult = false;
    //         console.log(this.document)
    //         var image = document.getElementById('output') ;
            
    //         (image as HTMLImageElement).src = URL.createObjectURL(files[fileindex]);
    //       }
    //     }
    //   }
            this.urls = [];
           
            if (files) {
              console.log(files)
              for (let fileindex=0 ; fileindex<Object.keys(files).length;fileindex++) {
                let reader = new FileReader();
                reader.onload = (e: any) => {
                  this.urls.push(e.target.result);
                  console.log(this.urls)
                }
                reader.readAsDataURL(files[fileindex]);
              }
            }
  
  
  
          
      
    
  }

}
