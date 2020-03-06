import { Component, OnInit } from '@angular/core';
import Chart from 'chart.js';
import { Router, ActivatedRoute } from '@angular/router';
import { ApiService } from '../api-service/api.service';
import { DataShareService } from '../data-share/data-share.service';
import { Ng4LoadingSpinnerService } from 'ng4-loading-spinner';
import Swal from 'sweetalert2';
import { VERSION } from '@angular/core';
import { HttpClientModule, HttpClient, HttpRequest, HttpResponse, HttpEventType } from '@angular/common/http';
import { SelectTrainModelComponent } from './select-train-model/select-train-model.component';

declare var $: any;

declare interface DataTable {
  headerRow: string[];
  // footerRow: string[];
  dataRows: string[];


}

@Component({
  moduleId: module.id,
  selector: 'show-templates',
  templateUrl: './show-templates.component.html',
  styleUrls: ['./show-templates.component.css'],
})
export class ShowTemplatesComponent implements OnInit {
  public loading = false;
  percentDone: number;
  select_; Train_;
  importFolder;
  import_: string = 'active'; test_; Eval_;
  modelSelect: boolean = false;
  trainSelect: boolean = false;
  importSelect: boolean = true;
  evaluateSelect: boolean = false;
  testSelect: boolean = false;
  uploadSuccess: boolean;
  objectKeys = Object.keys;
  displayModel: boolean = false;
  traverse:boolean=false;
  trainModelInput={};
  importClasses;
  importclassCount;
  stepsPerEpoch;
  selectModelName;
  modelTraining;
  editImport:boolean=false;
  epoch;learningRate;
  jsonStringify = JSON.stringify;
  public dataTable: DataTable = null;
  public API_DOCUMENTATION: string;
  constructor(
    private router: Router,
    private route: ActivatedRoute,
    private http: HttpClient,
    private apiservice: ApiService,
    private spinnerService: Ng4LoadingSpinnerService,
    private dataService: DataShareService,
  ) { }
  ngOnInit() {
    this.dataService.changeMessage("Train Model")
    // this.spinnerService.show();
    this.route.queryParamMap.subscribe(
      params => {
        // this.modelSelect = JSON.parse(params.get('modelSelect'));
        // this.trainSelect = JSON.parse(params.get('trainSelect'));
        // this.evaluateSelect = JSON.parse(params.get('evaluateSelect'));
        console.log(this.modelSelect)
        if (JSON.parse(params.get('modelSelect'))) {
          this.importclassCount = JSON.parse(params.get('importclassCount'));
          this.importClasses = JSON.parse(params.get('importClasses'));
          this.importFolder = JSON.parse(params.get('importFolder'));
          this.stepsPerEpoch = JSON.parse(params.get('stepsPerEpoch'));
          console.log(this.importFolder)
          this.trainModelInput["importClassCount"] = this.importclassCount
          this.trainModelInput["importClasses"] = this.importClasses
          this.trainModelInput["importFolder"] = this.importFolder
          this.trainModelInput["stepsPerEpoch"] = this.stepsPerEpoch

          
          this.traverse=true;       
          

        }
        else if (JSON.parse(params.get('trainSelect'))) {
          console.log("*************************************************************")
          this.selectModelName=JSON.parse(params.get('selectModelName'))
          this.trainModelInput["selectModelName"]=this.selectModelName
          this.traverse=true;
          
        }

        else if (JSON.parse(params.get('evaluateSelect'))) {
          this.traverse=true;
          this.epoch=JSON.parse(params.get('epochs'));
          this.learningRate=JSON.parse(params.get('learningRate'));
          this.modelTraining=JSON.parse(params.get('modelTraining'))
          this.stepsPerEpoch = JSON.parse(params.get('stepsPerEpoch'));
          this.trainModelInput["epochs"] = this.epoch;
          this.trainModelInput["learningRate"] = this.learningRate;
          this.trainModelInput["modelTraining"] = this.modelTraining;
          this.trainModelInput["stepsPerEpoch"] = this.stepsPerEpoch
          console.log("results from train component",this.trainModelInput)
          this.apiservice.getTrainResults(this.trainModelInput).subscribe(
            resp=>{
                console.log(resp)
            }
          );
        }

      },
      err => {
        //console.log(err);
      })


  }
  upload(files: File[]) {
    //pick from one of the 4 styles of file uploads below
    console.log(typeof files)
    for (let each_file = 0; each_file < files.length; each_file++) {
      console.log(files[0])
      console.log(typeof files)
    }
    this.uploadAndProgress(files);
  }

  basicUpload(files: File[]) {
    var formData = new FormData();
    Array.from(files).forEach(f => formData.append('file', f))
    console.log(formData);
    this.http.post('https://file.io', formData)
      .subscribe(event => {
        console.log('done')
      })
  }

  //this will fail since file.io dosen't accept this type of upload
  //but it is still possible to upload a file with this style
  basicUploadSingle(file: File) {
    this.http.post('https://file.io', file)
      .subscribe(event => {
        console.log('done')
      })
  }



  uploadAndProgress(files: File[]) {
    console.log(files)
    var formData = new FormData();
    Array.from(files).forEach(f => formData.append('file', f))

    this.http.post('https://file.io', formData, { reportProgress: true, observe: 'events' })
      .subscribe(event => {
        if (event.type === HttpEventType.UploadProgress) {
          this.percentDone = Math.round(100 * event.loaded / event.total);
        } else if (event instanceof HttpResponse) {
          this.uploadSuccess = true;
        }
        console.log(event)
      });
    this.basicUpload(files)
  }

  //this will fail since file.io dosen't accept this type of upload
  //but it is still possible to upload a file with this style
  uploadAndProgressSingle(file: File) {
    this.http.post('https://file.io', file, { reportProgress: true, observe: 'events' })
      .subscribe(event => {
        if (event.type === HttpEventType.UploadProgress) {
          this.percentDone = Math.round(100 * event.loaded / event.total);
        } else if (event instanceof HttpResponse) {
          this.uploadSuccess = true;
        }
      });
  }
  displayModels() {
    if (this.displayModel == false) {
      this.displayModel = true;
    }
    else {
      this.displayModel = false;
    }
  }

  ImportData() {
    // let route:boolean=false;
    // if (this.traverse && (this.trainSelect || this.modelSelect || this.evaluateSelect || this.testSelect)){
    //     route = true;
    //   }
    this.importSelect = true;
    this.trainSelect = false;
    this.modelSelect = false;
    this.evaluateSelect = false;
    this.testSelect = false;
    this.import_ = 'active';
    this.select_ = '';
    this.Train_ = '';
    this.test_ = '';
    this.Eval_ = '';

    document.getElementById("Import").setAttribute("active", "True");
    
    console.log(this.importclassCount)
    console.log(this.importClasses)
    // if(route==true){
    //   this.router.navigate(["show"])
    // }
  }
  ModelSelect() {
    // let route:boolean=false;
    // if (this.traverse && (this.trainSelect  || this.evaluateSelect || this.testSelect)){
    //     route = true;
    //   }
    console.log("value", this.modelSelect)
    this.modelSelect = true;
    this.trainSelect = false;
    this.importSelect = false;
    this.evaluateSelect = false;
    this.testSelect = false;
    this.import_ = '';
    this.select_ = 'active';
    this.Train_ = '';
    this.test_ = '';
    this.Eval_ = '';
    
    console.log(this.importclassCount)
    console.log(this.importClasses)
    this.trainModelInput["select"]='chooseModel'
    // if(route==true){
    //   this.router.navigate(["show"],{ queryParams: {'modelSelect': JSON.stringify(this.modelSelect)}})
    // }
  }
  TrainModel() {
    // let route:boolean=false;
    // if (this.traverse && (this.modelSelect  || this.evaluateSelect || this.testSelect)){
    //     route = true;
    //   }   
    this.trainSelect = true;
    this.importSelect = false;
    this.modelSelect = false;
    this.evaluateSelect = false;
    this.testSelect = false;
    this.import_ = '';
    this.test_ = '';
    this.Eval_ = '';
    this.Train_ = 'active';
    this.select_ = '';
   
    console.log(this.importclassCount)
    console.log(this.importClasses)
    this.trainModelInput["select"]='chooseTrain'
    // if(route==true){
    //   this.router.navigate(["show"],{ queryParams: {'trainSelect': JSON.stringify(true)} })
    // }
  }
  Evaluate() {
    // let route:boolean=false;
    // if (this.traverse && (this.modelSelect  || this.trainSelect || this.testSelect)){
    //     route = true;
    //   }
    this.trainSelect = false;
    this.importSelect = false;
    this.modelSelect = false;
    this.evaluateSelect = true;
    this.testSelect = false;
    this.import_ = '';
    this.test_ = '';
    this.Eval_ = 'active';
    this.Train_ = '';
    this.select_ = '';
    console.log(this.importclassCount)
    console.log(this.importClasses)
    
    // if(route==true){
    //   this.router.navigate(["show"],{ queryParams: {'evaluateSelect': JSON.stringify(true)} })
    // }
  }
  TestandDeploy() {
    // let route:boolean=false;
    // if (this.traverse && (this.modelSelect  || this.trainSelect || this.evaluateSelect)){
    //     route = true;
    //   }
    this.trainSelect = false;
    this.importSelect = false;
    this.modelSelect = false;
    this.evaluateSelect = false;
    this.testSelect = true;
    this.import_ = '';
    this.test_ = 'active';
    this.Eval_ = '';
    this.Train_ = '';
    this.select_ = '';
    console.log(this.importclassCount)
    console.log(this.importClasses)
    // if(route==true){
    //   this.router.navigate(["show"])
    // }
  }
  editImportModel() {
    this.importClasses = null;
    this.importclassCount = null;
    this.editImport=true;
  }
  editSelectModel() {
    this.selectModelName=null;
  }
}
