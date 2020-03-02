import { Component, OnInit ,Input} from '@angular/core';
import { SelectTrainModelComponent } from '../select-train-model/select-train-model.component';
@Component({
  selector: 'app-test-evaluate-model',
  templateUrl: './test-evaluate-model.component.html',
  styleUrls: ['./test-evaluate-model.component.css'],
  providers: [SelectTrainModelComponent]
})
export class TestEvaluateModelComponent implements OnInit {
  @Input() header:string;
  constructor(private selectTrain: SelectTrainModelComponent) { }

  ngOnInit() {
    console.log(this.header)
    console.log(this.selectTrain.globalStr)
    this.selectTrain.globalStr="changed text"
  }

}
