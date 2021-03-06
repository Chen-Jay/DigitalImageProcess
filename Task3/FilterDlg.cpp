// FilterDlg.cpp: 实现文件
//

#include "stdafx.h"
#include "Task3.h"
#include "FilterDlg.h"
#include "afxdialogex.h"


// FilterDlg 对话框

IMPLEMENT_DYNAMIC(FilterDlg, CDialog)

FilterDlg::FilterDlg(CWnd* pParent /*=nullptr*/)
	: CDialog(IDD_FILTER, pParent)
{

}

FilterDlg::~FilterDlg()
{
}

BOOL FilterDlg::OnInitDialog()
{
	CDialog::OnInitDialog();
	mFilterSelect.InsertString(0, _T("平滑线性滤波"));
	mFilterSelect.InsertString(1, _T("高斯滤波"));
	mFilterSelect.InsertString(2, _T("维纳滤波"));
	mFilterSelect.SetCurSel(0);

	mFilterVariance.SetReadOnly(true);

	return 0;
}

void FilterDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_FILTERSELECT, mFilterSelect);
	//DDX_Control(pDX, IDC_FILTER_MEANS, mFilterMeans);
	DDX_Control(pDX, IDC_FILTER_VARIANCE, mFilterVariance);
}


BEGIN_MESSAGE_MAP(FilterDlg, CDialog)
	ON_CBN_SELCHANGE(IDC_FILTERSELECT, &FilterDlg::OnCbnSelchangeFilterselect)
END_MESSAGE_MAP()


// FilterDlg 消息处理程序


void FilterDlg::OnCbnSelchangeFilterselect()
{
	// TODO: 在此添加控件通知处理程序代码
	if (mFilterSelect.GetCurSel() == 1)
	{
		mFilterVariance.SetReadOnly(false);
	}
	else 
	{
		mFilterVariance.SetReadOnly(true);
	}

}
