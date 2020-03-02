import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SelectTrainModelComponent } from './select-train-model.component';

describe('SelectTrainModelComponent', () => {
  let component: SelectTrainModelComponent;
  let fixture: ComponentFixture<SelectTrainModelComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SelectTrainModelComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SelectTrainModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
