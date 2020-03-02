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
  fileName: string;
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
    const formData: FormData = new FormData();
    formData.append('file', files[0], files[0].name);
    this.files = files;
    this.fileName = files[0].name
    this.extension = this.fileName.split(".")[1]
    
    
    // this.changeFile=false
    this.apiservice.getUnzippedFiles(formData).subscribe(
      success => {
        
        console.log("result from backend :",success);
        
      }
      
      
    );
    
  if (files && files[0]) {
    const file = files[0];
    this.fileToUpload = files[0];
    this.fileName = this.fileToUpload.name;
    const reader = new FileReader();
    reader.onload = e => (this.document = reader.result);
    reader.readAsDataURL(file);
    this.showResult = false;
    console.log(this.document)
    
  }
}

}
