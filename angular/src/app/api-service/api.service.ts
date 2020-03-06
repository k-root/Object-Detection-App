import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { map } from 'rxjs/operators';
import { HttpClient, HttpRequest, HttpHeaders } from '@angular/common/http';
import { ResponseType } from '@angular/http';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  // BASE_URL: string = window.location.pathname;
  BASE_URL: string = ''
  // BASE_URL: string = 'http://localhost:8080'
  API_URL = `${this.BASE_URL}/api`;
  SAMPLE_IMAGES_URL = `${this.API_URL}/sample`
  INTEGRATION_URL = `${this.API_URL}/integration`;
  TEMPLATES_URL = `${this.API_URL}/templates`;
  TRMPLATE_URL = `${this.API_URL}/template`;
  TEST_TEMPLATE = `${this.TRMPLATE_URL}/test`;
  API_DOCUMENTATION = `${this.API_URL}/`;
  IMAGE_CONVERT = `${this.BASE_URL}/convert/`;
  IMPORT_URL = `${this.API_URL}/template/import`;
  PROCESSOE = `${this.API_URL}/processors`;
  TEMPLATE_TYPE_LIST = `${this.API_URL}/templates/types`;
  GENERIC_TEMP_LIST = `${this.API_URL}/templates/generic`;
  INTEGRATION_BASE_URL = `${this.API_URL}/integration`;
  TARGET_CONNECTIONS_LIST = `${this.INTEGRATION_BASE_URL}/targetconnection/list`;
  SOURCE_CONNECTIONS_LIST = `${this.INTEGRATION_BASE_URL}/sourceconnection/list`;
  TARGET_CONNECTION_ADD = `${this.INTEGRATION_BASE_URL}/targetconnection/add-update`;
  SOURCE_CONNECTION_ADD = `${this.INTEGRATION_BASE_URL}/sourceconnection/add-update`;
  TARGET_CONNECTION_GET = `${this.INTEGRATION_BASE_URL}/targetconnection/get`;
  SOURCE_CONNECTION_GET = `${this.INTEGRATION_BASE_URL}/sourceconnection/get`;
  FLOWS_LIST = `${this.INTEGRATION_BASE_URL}/list`;
  FLOWS_ADD_UPDATE = `${this.INTEGRATION_BASE_URL}/add-update`;
  FLOW_GET = `${this.INTEGRATION_BASE_URL}/get`;
  FLOW_BASE = `${this.API_URL}/flows`;
  RUN_FLOW = `${this.FLOW_BASE}/execute/job`;
  QUEUED_JOBS = `${this.FLOW_BASE}/queuedjobs`;
  COMPLETED_JOBS = `${this.FLOW_BASE}/completedjobs`;
  JOB_RESULT = `${this.FLOW_BASE}/task/doc/output`;
  JOB_INPUT = `${this.FLOW_BASE}/task/doc/input`;
  JOB_OUTPUT_UPDATE = `${this.FLOW_BASE}/job-output/update`;
  JOB_SUBMIT_TARGET = `${this.FLOW_BASE}/job-output/submit-to-target`;
  ENDPOINT_TEMPLATES = `${this.INTEGRATION_BASE_URL}/endpoints`

  LIST_SAMPLE_IMAGES =  `${this.API_URL}/sample/get-images`
  SAMPLE_IMAGE_URL = `${this.API_URL}/sample/load-image`


  LEARN_TEMPLATE_URL = `${this.API_URL}/learn/learntTemplate`

  username:string;
  headerDict:any;
  requestOptions:any;

  constructor(private http: HttpClient) {
  }


  updateLearnTemplate(templateName, learnTemplateJson):Observable<any>{
    this.init();
    return this.http.post(`${this.LEARN_TEMPLATE_URL}/${templateName}`,learnTemplateJson, this.requestOptions);
  }


  getLearnTemplate(templateName):Observable<any>{
    this.init();
    return this.http.get(`${this.LEARN_TEMPLATE_URL}/${templateName}`, this.requestOptions);
  }

  getUnzippedFiles(formData){
    this.init();
    return this.http.post(`${this.BASE_URL}/api/zipfile`, formData,this.requestOptions);
  }

  sendImages(formData){
    this.init();
    console.log(formData)
    return this.http.post(`${this.BASE_URL}/api/testImages`, formData,this.requestOptions);
  }

  getResultImages(){
    this.init();
    return this.http.get(`${this.BASE_URL}/api/getResultImages`, this.requestOptions)
  }

  getModelNames():Observable<any>{
    this.init();
    return this.http.get(`${this.BASE_URL}/api/getModelNames`,this.requestOptions);
  }

  getTrainResults(modelInput):Observable<any>{
    this.init();
    console.log(modelInput, "entered train result")
    return this.http.post(`${this.BASE_URL}/api/train`,{modelInput},this.requestOptions);
  }

  getSampleImagesByType(templateType):Observable<any>{
    this.init();
    return this.http.post(`${this.LIST_SAMPLE_IMAGES}`, {'templateType':templateType}, this.requestOptions)
  }

  getSampleImage(templateType, imageName):Observable<any> {
    this.init();
    return this.http.post(`${this.SAMPLE_IMAGE_URL}`, { 'templateType':templateType, 'imageName':imageName }, {
      responseType: 'blob',
      headers: new HttpHeaders(this.headerDict)
    });
  }
  getTemplates(): Observable<any> {
    this.init();
    return this.http
      .get(this.TEMPLATES_URL, this.requestOptions)
      .pipe(map((response: any) => response));
  }
  setTemplate(files:File[]): Observable<any> {
    this.init();
    return this.http.post(`${this.TRMPLATE_URL}/unzipfile`, files, this.requestOptions);
  }
  getTemplateData(templateName): Observable<any> {
    this.init();
    return this.http.get(`${this.TRMPLATE_URL}/${templateName}`, this.requestOptions);
  }
  deleteTemplate(templateName): Observable<any> {
    this.init();
    return this.http.delete(`${this.TRMPLATE_URL}/${templateName}`, this.requestOptions);
  }
  test(templateName, formData): Observable<any> {
    this.init();
    return this.http.post(`${this.TEST_TEMPLATE}/${templateName}`, formData, this.requestOptions);
  }
  convertImage(formData): Observable<any> {
    this.init();
    return this.http.post(this.IMAGE_CONVERT, formData, {
      responseType: 'blob'
    });
  }
  integration(data):Observable<any> {
    this.init();
    return this.http.post(this.INTEGRATION_URL, data, this.requestOptions);
  }
  importTemplate(templateName):Observable<any> {
    this.init();
    return this.http.get(`${this.IMPORT_URL}/${templateName}`, this.requestOptions);
  }
  getProcessor(): Observable<any>  {
    return this.http.get(this.PROCESSOE);
  }
  getGenericTemplates(): Observable<any> {
    return this.http.get(this.GENERIC_TEMP_LIST);
  }
  xlsToImage(formData): Observable<any> {
    this.init();
    return this.http.post('http://35.239.45.134:8080', formData, {
      responseType: 'blob'
    });
  }
  getTargetConnections(payload): Observable<any> {
    return this.http.post(this.TARGET_CONNECTIONS_LIST, payload);
  }
  getSourceConnections(payload): Observable<any> {
    return this.http.post(this.SOURCE_CONNECTIONS_LIST, payload);
  }
  addSourceConnection(payload): Observable<any> {
    return this.http.post(this.SOURCE_CONNECTION_ADD, payload);
  }
  getSourceConnection(payload): Observable<any> {
    return this.http.post(this.SOURCE_CONNECTION_GET, payload);
  }
  addTargetConnection(payload): Observable<any> {
    return this.http.post(this.TARGET_CONNECTION_ADD, payload);
  }
  getTargetConnection(payload): Observable<any> {
    return this.http.post(this.TARGET_CONNECTION_GET, payload);
  }
  getFlows(payload): Observable<any> {
    return this.http.post(this.FLOWS_LIST, payload);
  }
  flowAddUpdate(payload): Observable<any> {
    return this.http.post(this.FLOWS_ADD_UPDATE, payload);
  }
  getFlow(payload): Observable<any> {
    return this.http.post(this.FLOW_GET, payload);
  }
  runFlow(payload): Observable<any> {
    return this.http.post(this.RUN_FLOW, payload);
  }
  queuedjobs(payload): Observable<any> {
    return this.http.post(this.QUEUED_JOBS, payload);
  }
  completedJobs(payload): Observable<any> {
    return this.http.post(this.COMPLETED_JOBS, payload);
  }
  getJobResult(payload): Observable<any> {
    return this.http.post(this.JOB_RESULT, payload);
  }
  getJobInput(payload): Observable<any> {
    return this.http.post(this.JOB_INPUT, payload, {
      responseType: 'blob'
    });
  }
  updateJobOutput(payload): Observable<any> {
    return this.http.post(this.JOB_OUTPUT_UPDATE,payload)
  }
  submitToTarget(payload): Observable<any>{
    return this.http.post(this.JOB_SUBMIT_TARGET,payload)
  }
  getEnpointTemplates(payload): Observable<any>{
    return this.http.post(this.ENDPOINT_TEMPLATES,payload)
  }
  getTemplateTypes(): Observable<any>{
    this.init();
    return this.http.get(this.TEMPLATE_TYPE_LIST, this.requestOptions);
  }
  init(){
    this.username = localStorage.getItem('username');
    if(!this.username){this.username='user'};
    this.headerDict = {
      username: this.username
    };
    this.requestOptions = {
      headers: new HttpHeaders(this.headerDict)
    };
  }
}
